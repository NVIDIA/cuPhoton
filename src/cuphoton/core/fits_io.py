# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""FITS image ingress shared by CPU and GPU workflows.

``auto`` uses xDR for supported lossless images when its GPU dependencies
are available. Unsupported FITS scaling, nulls and compression use Astropy.
Once payload reading starts, errors propagate; a failed GPU read is never
silently retried with different semantics. Choosing xDR does not assert that
the storage route is native GPUDirect Storage.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

from .fits_options import normalize_xdr_options

_BITPIX_DTYPES = {8: "u1", 16: "i2", 32: "i4", 64: "i8", -32: "f4", -64: "f8"}


@dataclass(frozen=True)
class FitsImageInfo:
    """Logical image metadata obtained without decoding any image pixels."""

    path: Path
    hdu: int
    shape: tuple[int, int]
    dtype: np.dtype
    header: fits.Header
    compression: str | None
    xdr_unsupported_reason: str | None


@dataclass(frozen=True)
class FitsReadResult:
    arrays: tuple[Any, ...]
    infos: tuple[FitsImageInfo, ...]
    reader: str
    requested_reader: str
    fallback_reason: str | None
    device: bool
    xdr_options: Mapping[str, str] | None = None

    def metadata(self) -> dict[str, Any]:
        """Return logical read provenance, without physical I/O claims."""
        return {
            "reader": self.reader,
            "requested_reader": self.requested_reader,
            "location": "device" if self.device else "host",
            "fallback_reason": self.fallback_reason,
            "hdus": [
                {
                    "hdu": info.hdu,
                    "shape": list(array.shape),
                    "dtype": array.dtype.str,
                    "compression": info.compression,
                }
                for info, array in zip(self.infos, self.arrays, strict=True)
            ],
            "decoded_bytes": sum(int(array.nbytes) for array in self.arrays),
            **(
                {"xdr_options": dict(self.xdr_options)}
                if self.xdr_options
                else {}
            ),
        }


def _logical_dtype(header: fits.Header) -> np.dtype:
    bitpix = int(header["BITPIX"])
    dtype = np.dtype(_BITPIX_DTYPES[bitpix])
    scale, zero = header.get("BSCALE", 1), header.get("BZERO", 0)
    if scale == 1:
        if bitpix == 8 and zero == -128:
            return np.dtype("i1")
        if bitpix in (16, 32, 64) and zero == 1 << (bitpix - 1):
            return np.dtype(f"u{bitpix // 8}")
    if bitpix > 0 and (
        scale != 1 or zero != 0 or header.get("BLANK") is not None
    ):
        return np.dtype("f8" if bitpix > 16 else "f4")
    return dtype


def _image_info(path: Path, index: int, hdu: Any) -> FitsImageInfo | None:
    if not isinstance(
        hdu, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)
    ):
        return None
    header = hdu.header.copy()
    if header.get("NAXIS") != 2:
        return None
    shape = (int(header["NAXIS2"]), int(header["NAXIS1"]))
    if min(shape) <= 0:
        return None
    compression = None
    reason = None
    if isinstance(hdu, fits.CompImageHDU):
        # The logical header hides the compression-table representation.
        # Inspect only its header, never the table or image .data property.
        physical = hdu._bintable.header
        compression = str(physical["ZCMPTYPE"]).strip().upper()
        columns = {
            str(physical.get(f"TTYPE{i}", "")).strip().upper()
            for i in range(1, int(physical.get("TFIELDS", 0)) + 1)
        }
        if compression not in {"GZIP_1", "GZIP_2"}:
            reason = f"compression {compression} requires Astropy"
        elif columns & {"ZSCALE", "ZZERO", "ZBLANK"}:
            reason = "quantized FITS images require Astropy"
        elif physical.get("ZBLANK") is not None:
            reason = "FITS null values require Astropy"
    if header.get("BLANK") is not None:
        reason = "FITS BLANK values require Astropy"
    elif header.get("BSCALE", 1) != 1 or header.get("BZERO", 0) != 0:
        reason = "FITS BSCALE/BZERO scaling requires Astropy"
    if getattr(hdu._file, "compression", None):
        reason = "externally compressed FITS files require Astropy"
    return FitsImageInfo(
        path,
        index,
        shape,
        _logical_dtype(header),
        header,
        compression,
        reason,
    )


def inspect_fits_images(path: str | Path) -> tuple[FitsImageInfo, ...]:
    """List two-dimensional image HDUs without reading their pixels."""
    resolved = Path(path).expanduser().resolve()
    with fits.open(resolved, memmap=False, lazy_load_hdus=True) as hdus:
        return tuple(
            info
            for index, hdu in enumerate(hdus)
            if (info := _image_info(resolved, index, hdu)) is not None
        )


def inspect_fits_image(
    path: str | Path, hdu: int | None = None
) -> FitsImageInfo:
    """Select one image, defaulting to the first two-dimensional image."""
    if hdu is not None and (
        isinstance(hdu, bool)
        or not isinstance(hdu, (int, np.integer))
        or hdu < 0
    ):
        raise ValueError("FITS HDU must be a nonnegative integer")
    resolved = Path(path).expanduser().resolve()
    with fits.open(resolved, memmap=False, lazy_load_hdus=True) as hdus:
        if hdu is not None:
            try:
                selected = hdus[int(hdu)]
            except IndexError as exc:
                raise ValueError(f"FITS HDU {hdu} is out of range") from exc
            info = _image_info(resolved, int(hdu), selected)
            if info is None:
                raise ValueError(f"FITS HDU {hdu} is not a 2D image")
            return info
        for index, selected in enumerate(hdus):
            info = _image_info(resolved, index, selected)
            if info is not None:
                return info
    raise ValueError(f"No 2D image HDU found in {resolved}")


def _validate_section(section: Any, infos: Sequence[FitsImageInfo]):
    if section is None:
        return None
    if not isinstance(section, tuple) or len(section) != 2:
        raise ValueError("FITS section must be a pair of slices")
    for axis, part in enumerate(section):
        if not isinstance(part, slice) or part.step not in (None, 1):
            raise ValueError("FITS section requires unit-step slices")
        for info in infos:
            start = 0 if part.start is None else part.start
            stop = info.shape[axis] if part.stop is None else part.stop
            if (
                isinstance(start, bool)
                or isinstance(stop, bool)
                or not isinstance(start, (int, np.integer))
                or not isinstance(stop, (int, np.integer))
                or not 0 <= start < stop <= info.shape[axis]
            ):
                raise ValueError("FITS section must lie within every image")
    return section


def _xdr_available() -> bool:
    from cuphoton.xdr import gpu_available
    from cuphoton.xdr.nvcomp_batch import native_plan_files_available

    return gpu_available() and native_plan_files_available()


def _read_xdr(
    path: Path, hdus: tuple[int, ...], *, section, stream, xdr_options
):
    from cuphoton.xdr import batch_to_device_stream

    return batch_to_device_stream(
        [path],
        hdus,
        section=section,
        stream=stream,
        prefetch_depth=1,
        decode_batch_files=1,
        batch_queue_depth=1,
        native_read_threads=1,
        native_plan_threads=1,
        **xdr_options,
    )


def validate_fits_reader(reader: str) -> str:
    """Validate one explicit FITS reader policy without importing CUDA."""
    if not isinstance(reader, str) or reader not in {
        "astropy",
        "auto",
        "xdr",
    }:
        raise ValueError("FITS reader must be astropy, auto, or xdr")
    return reader


def read_fits_images(
    path: str | Path,
    hdus: Sequence[int],
    *,
    reader: str = "astropy",
    device: bool = False,
    section=None,
    stream=None,
    xdr_options: Mapping[str, str] | None = None,
) -> FitsReadResult:
    """Read selected planes to host or device, preserving FITS semantics.

    Arrays use native byte order. A section is a bounded ``(y, x)`` pair of
    unit-step slices. Uncompressed sections use Astropy under ``auto`` so a
    small stamp does not require a full-image GPU read. Explicit ``xdr``
    rejects unsupported semantics or unavailable dependencies before reading
    pixels. All returned device arrays are ready on return.

    ``xdr_options`` accepts ``postprocess`` (auto/fused/separate),
    ``gzip_decoder`` (auto/gzip/deflate), and ``decompression_backend``
    (auto/cuda). Omitted choices use xDR's ``auto`` defaults. For example,
    ``xdr_options={"decompression_backend": "cuda"}`` requests CUDA kernels
    when xDR decodes a compressed HDU; it requires a native helper with
    backend selection. These choices do not change the reader or its Astropy
    fallback policy. Metadata records explicit choices as requested settings,
    including when Astropy is selected; it does not identify an nvCOMP engine.
    """
    validate_fits_reader(reader)
    xdr_options = normalize_xdr_options(xdr_options)
    selectors = tuple(hdus)
    if not selectors:
        raise ValueError("At least one FITS HDU must be selected")
    # Only headers are accessed before choosing a reader.
    infos = tuple(inspect_fits_image(path, hdu=hdu) for hdu in selectors)
    section = _validate_section(section, infos)
    reason = next(
        (
            info.xdr_unsupported_reason
            for info in infos
            if info.xdr_unsupported_reason
        ),
        None,
    )
    if section is not None and any(
        info.compression is None for info in infos
    ):
        reason = "uncompressed FITS sections require Astropy"
    actual = "astropy"
    fallback = None
    if reader != "astropy":
        if reason is None and not _xdr_available():
            reason = "xDR GPU dependencies or a CUDA device are unavailable"
        if reason is not None:
            if reader == "xdr":
                raise ValueError(f"Cannot use xDR: {reason}")
            fallback = reason
        else:
            actual = "xdr"
    resolved = infos[0].path
    unique = tuple(dict.fromkeys(int(hdu) for hdu in selectors))
    if actual == "xdr":
        stacked = _read_xdr(
            resolved,
            unique,
            section=section,
            stream=stream,
            xdr_options=xdr_options,
        )
        if len(stacked) != len(unique):
            raise RuntimeError("xDR returned different HDU coverage")
        by_hdu = dict(
            zip(unique, (array[0] for array in stacked), strict=True)
        )
        if not device:
            import cupy as cp

            by_hdu = {hdu: cp.asnumpy(array) for hdu, array in by_hdu.items()}
    else:
        by_hdu = {}
        with fits.open(resolved, memmap=False) as source:
            for hdu in unique:
                data = (
                    source[hdu].data
                    if section is None
                    else source[hdu].section[section]
                )
                array = np.asarray(data)
                # Own the result after closing the FITS file and normalize
                # byte order without a float conversion of integer masks.
                by_hdu[hdu] = np.array(
                    array,
                    dtype=array.dtype.newbyteorder("="),
                    copy=True,
                )
        if device:
            import cupy as cp

            active_stream = (
                stream if stream is not None else cp.cuda.get_current_stream()
            )
            with active_stream:
                by_hdu = {
                    hdu: cp.asarray(array, blocking=True)
                    for hdu, array in by_hdu.items()
                }
            active_stream.synchronize()
    arrays = tuple(by_hdu[int(hdu)] for hdu in selectors)
    for array, info in zip(arrays, infos, strict=True):
        expected_shape = (
            info.shape
            if section is None
            else tuple(
                len(range(*part.indices(size)))
                for part, size in zip(section, info.shape, strict=True)
            )
        )
        if tuple(array.shape) != expected_shape or array.dtype != info.dtype:
            raise RuntimeError(
                "FITS image shape or dtype changed during read"
            )
    return FitsReadResult(
        arrays, infos, actual, reader, fallback, device, xdr_options
    )
