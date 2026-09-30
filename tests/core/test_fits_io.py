# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import gzip
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits

from cuphoton.core import fits_io


@pytest.mark.parametrize("decompression_backend", ["auto", "cuda"])
@pytest.mark.parametrize("postprocess", ["fused", "separate"])
@pytest.mark.parametrize("gzip_decoder", ["auto", "gzip", "deflate"])
def test_xdr_options_reach_batch_dispatch(
    tmp_path, monkeypatch, postprocess, gzip_decoder, decompression_backend
):
    import cuphoton.xdr as xdr

    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    path = _write(tmp_path, data, compression="GZIP_2")
    observed = {}

    def batch(paths, hdus, **options):
        observed.update(options)
        return [data[None]]

    monkeypatch.setattr(fits_io, "_xdr_available", lambda: True)
    monkeypatch.setattr(xdr, "batch_to_device_stream", batch)
    result = fits_io.read_fits_images(
        path,
        [1],
        reader="xdr",
        device=True,
        xdr_options={
            "postprocess": postprocess,
            "gzip_decoder": gzip_decoder,
            "decompression_backend": decompression_backend,
        },
    )
    assert observed["postprocess"] == postprocess
    assert observed["gzip_decoder"] == gzip_decoder
    assert observed["decompression_backend"] == decompression_backend
    assert result.metadata()["xdr_options"] == {
        "postprocess": postprocess,
        "gzip_decoder": gzip_decoder,
        "decompression_backend": decompression_backend,
    }
    np.testing.assert_array_equal(result.arrays[0], data)


@pytest.mark.parametrize(
    "options", ["fused", {"unknown": "auto"}, {"postprocess": True}]
)
def test_invalid_xdr_options_fail_before_input_access(tmp_path, options):
    with pytest.raises(ValueError, match="xDR|xdr_options"):
        fits_io.read_fits_images(
            tmp_path / "missing.fits", [1], xdr_options=options
        )


def _write(tmp_path, array, *, compression=None, **kwargs):
    path = tmp_path / "images.fits"
    hdu = (
        fits.ImageHDU(array, **kwargs)
        if compression is None
        else fits.CompImageHDU(
            array, compression_type=compression, quantize_level=0, **kwargs
        )
    )
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
    return path


@pytest.mark.parametrize("compression", [None, "GZIP_1", "GZIP_2"])
@pytest.mark.parametrize(
    "dtype", ["i2", "i4", "i8", "u2", "u4", "u8", "f4", "f8"]
)
def test_host_matches_astropy_dtype_bits_and_section(
    tmp_path, compression, dtype
):
    array = np.arange(35, dtype=dtype).reshape(5, 7)
    if array.dtype.kind == "f":
        array[0, 0], array[0, 1] = -0.0, np.nan
    if dtype == "u8":
        array[-1, -1] = 2**63 + 17
    path = _write(tmp_path, array, compression=compression)
    info = fits_io.inspect_fits_image(path)
    with fits.open(path, memmap=False) as source:
        oracle = np.array(
            source[1].data, dtype=source[1].data.dtype.newbyteorder("=")
        )
    result = fits_io.read_fits_images(path, [1, 1])
    assert info.dtype == oracle.dtype
    assert result.arrays[0].tobytes() == oracle.tobytes()
    assert result.arrays[0] is result.arrays[1]
    section = (slice(1, 4), slice(2, 6))
    crop = fits_io.read_fits_images(path, [1], section=section)
    assert crop.arrays[0].tobytes() == oracle[section].tobytes()
    assert crop.metadata()["hdus"][0]["shape"] == [3, 4]


@pytest.mark.parametrize("bitpix", [8, 16, 32, 64, -32, -64])
def test_scaled_images_keep_astropy_semantics(tmp_path, bitpix):
    dtype = fits_io._BITPIX_DTYPES[bitpix]
    data = np.arange(12, dtype=dtype).reshape(3, 4)
    header = fits.Header({"BSCALE": 1.125, "BZERO": 128.25})
    path = _write(tmp_path, data, header=header)
    # Astropy constructors normalize scaling cards; set the stored header.
    with fits.open(
        path, mode="update", do_not_scale_image_data=True
    ) as source:
        source[1].header["BSCALE"] = 1.125
        source[1].header["BZERO"] = 128.25
    result = fits_io.read_fits_images(path, [1], reader="auto")
    with fits.open(path, memmap=False) as source:
        oracle = source[1].data
    np.testing.assert_array_equal(result.arrays[0], oracle)
    assert result.arrays[0].dtype == oracle.dtype.newbyteorder("=")
    assert result.reader == "astropy"
    assert "scaling" in result.fallback_reason


@pytest.mark.parametrize("dtype", ["i2", "i4", "i8"])
def test_blank_pixels_fall_back_before_gpu_probe(
    tmp_path, monkeypatch, dtype
):
    path = _write(tmp_path, np.arange(12, dtype=dtype).reshape(3, 4))
    with fits.open(path, mode="update") as source:
        source[1].header["BLANK"] = 3
    monkeypatch.setattr(
        fits_io, "_xdr_available", lambda: pytest.fail("GPU probe")
    )
    result = fits_io.read_fits_images(path, [1], reader="auto")
    assert np.isnan(result.arrays[0][0, 3])
    assert "BLANK" in result.fallback_reason
    with pytest.raises(ValueError, match="BLANK"):
        fits_io.read_fits_images(path, [1], reader="xdr")


def test_metadata_never_accesses_pixels_or_table_data(tmp_path, monkeypatch):
    path = _write(tmp_path, np.ones((3, 4), dtype="f4"), compression="GZIP_2")

    def forbidden(*args):
        pytest.fail("metadata inspection decoded pixels")

    monkeypatch.setattr(fits.CompImageHDU, "data", property(forbidden))
    monkeypatch.setattr(fits.BinTableHDU, "data", property(forbidden))
    (info,) = fits_io.inspect_fits_images(path)
    assert info.hdu == 1 and info.shape == (3, 4)
    assert info.compression == "GZIP_2"
    assert info.xdr_unsupported_reason is None


def test_cpu_reader_does_not_import_gpu_dependencies(tmp_path, monkeypatch):
    path = _write(tmp_path, np.ones((3, 4), dtype="i4"))
    for name in ("cupy", "kvikio", "nvidia.nvcomp", "cuphoton.xdr"):
        monkeypatch.setitem(sys.modules, name, None)
    assert fits_io.read_fits_images(path, [1]).reader == "astropy"


def test_explicit_selection_rejects_table_empty_and_invalid_hdu(tmp_path):
    path = tmp_path / "images.fits"
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.BinTableHDU.from_columns([]),
            fits.ImageHDU(np.ones((3, 4), dtype="f4")),
        ]
    ).writeto(path)
    assert fits_io.inspect_fits_image(path).hdu == 2
    for hdu in (True, -1, 1.5, 0, 1, 9):
        with pytest.raises(ValueError):
            fits_io.inspect_fits_image(path, hdu=hdu)


@pytest.mark.parametrize(
    "section",
    [
        (slice(-1, 3), slice(None)),
        (slice(0, 99), slice(None)),
        (slice(2, 2), slice(None)),
        (slice(None, None, 2), slice(None)),
        (slice(None),),
    ],
)
def test_invalid_sections_reject_before_read(tmp_path, section):
    path = _write(tmp_path, np.ones((3, 4), dtype="f4"))
    with pytest.raises(ValueError, match="section"):
        fits_io.read_fits_images(path, [1], section=section)


def test_auto_does_not_retry_failed_gpu_payload(tmp_path, monkeypatch):
    path = _write(tmp_path, np.ones((3, 4), dtype="f4"))
    monkeypatch.setattr(fits_io, "_xdr_available", lambda: True)

    def fail(*args, **kwargs):
        raise OSError("short payload read")

    monkeypatch.setattr(fits_io, "_read_xdr", fail)
    with pytest.raises(OSError, match="short payload"):
        fits_io.read_fits_images(path, [1], reader="auto")


def test_xdr_preserves_selector_order_and_reports_host_transfer(
    tmp_path, monkeypatch
):
    data = np.arange(12, dtype="i4").reshape(3, 4)
    path = _write(tmp_path, data)
    monkeypatch.setattr(fits_io, "_xdr_available", lambda: True)
    monkeypatch.setattr(
        fits_io, "_read_xdr", lambda *args, **kwargs: [data[None]]
    )
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(asnumpy=np.copy))
    result = fits_io.read_fits_images(path, [1, 1], reader="xdr")
    assert result.reader == "xdr" and result.device is False
    np.testing.assert_array_equal(result.arrays[1], data)
    assert result.metadata()["decoded_bytes"] == 2 * data.nbytes
    assert result.metadata()["location"] == "host"


def test_rice_and_quantization_require_astropy(tmp_path, monkeypatch):
    path = _write(
        tmp_path,
        np.arange(12, dtype="i4").reshape(3, 4),
        compression="RICE_1",
    )
    monkeypatch.setattr(
        fits_io, "_xdr_available", lambda: pytest.fail("GPU probe")
    )
    result = fits_io.read_fits_images(path, [1], reader="auto")
    assert "RICE_1" in result.fallback_reason
    path.unlink()
    data = np.random.default_rng(1).normal(size=(40, 40)).astype("f4")
    fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(data)]).writeto(path)
    info = fits_io.inspect_fits_image(path)
    assert info.xdr_unsupported_reason is not None


def test_outer_compression_is_detected_from_file_not_suffix(
    tmp_path, monkeypatch
):
    path = _write(tmp_path, np.arange(12, dtype="i4").reshape(3, 4))
    payload = gzip.compress(path.read_bytes())
    path.write_bytes(payload)
    monkeypatch.setattr(
        fits_io, "_xdr_available", lambda: pytest.fail("GPU probe")
    )
    result = fits_io.read_fits_images(path, [1], reader="auto")
    assert result.reader == "astropy"
    assert "externally compressed" in result.fallback_reason
    with pytest.raises(ValueError, match="externally compressed"):
        fits_io.read_fits_images(path, [1], reader="xdr")


def test_auto_requires_native_planner_before_reading(tmp_path, monkeypatch):
    path = _write(tmp_path, np.ones((3, 4), dtype="f4"))
    monkeypatch.setitem(
        sys.modules,
        "cuphoton.xdr",
        SimpleNamespace(gpu_available=lambda: True),
    )
    monkeypatch.setitem(
        sys.modules,
        "cuphoton.xdr.nvcomp_batch",
        SimpleNamespace(native_plan_files_available=lambda: False),
    )
    result = fits_io.read_fits_images(path, [1], reader="auto")
    assert result.reader == "astropy"
    assert "unavailable" in result.fallback_reason
