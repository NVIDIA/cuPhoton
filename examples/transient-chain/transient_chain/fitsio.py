# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""FITS plane ingress with a selectable reader: ``auto``, ``xdr`` or
``astropy``.

``auto`` is the default and picks the best path the environment supports for
each file: the
xDataReader GPU decoder when the image is a lossless GZIP tile-compressed or
raw image and the
GPU dependencies are present, Astropy otherwise. ``xdr`` insists on the GPU
decoder and fails
before reading pixels when a file cannot take it. ``astropy`` decodes on the
host and copies to
the device. The decision is made by cuPhoton's shared reader
(``cuphoton.core.fits_io``,
0.1.3 and later); on older trees a reduced local fallback applies the same
rule.

Every read returns a receipt (reader used, fallback reason, HDUs, dtypes,
bytes) so a run's
provenance states which path it took.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:  # cuPhoton >= 0.1.3
    from cuphoton.core import fits_io as _fits_io
except ImportError:  # pragma: no cover - rc0 trees
    _fits_io = None


@dataclass(frozen=True)
class PlaneInfo:
    path: Path
    hdu: int
    shape: tuple[int, int]
    dtype: np.dtype
    header: Any
    compression: str | None
    xdr_unsupported_reason: str | None


def _xdr_available() -> bool:
    try:
        from cuphoton.xdr import gpu_available
        from cuphoton.xdr.nvcomp_batch import native_plan_files_available
    except ImportError:
        return False
    return bool(gpu_available() and native_plan_files_available())


def _inspect_fallback(path: Path, hdu: int | None) -> PlaneInfo:
    from astropy.io import fits

    with fits.open(path, memmap=False, lazy_load_hdus=True) as hdus:
        indices = [hdu] if hdu is not None else range(len(hdus))
        for index in indices:
            h = hdus[index]
            if (
                not isinstance(
                    h, (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)
                )
                or h.header.get("NAXIS") != 2
            ):
                continue
            header = h.header.copy()
            comp = None
            reason = None
            if isinstance(h, fits.CompImageHDU):
                comp = str(h._bintable.header["ZCMPTYPE"]).strip().upper()
                if comp not in {"GZIP_1", "GZIP_2"}:
                    reason = f"compression {comp} requires Astropy"
            if (
                header.get("BSCALE", 1) != 1
                or header.get("BZERO", 0) != 0
                or header.get("BLANK") is not None
            ):
                reason = "FITS scaling or BLANK values require Astropy"
            bitpix = int(header["BITPIX"])
            dtype = np.dtype(
                {8: "u1", 16: "i2", 32: "i4", 64: "i8", -32: "f4", -64: "f8"}[
                    bitpix
                ]
            )
            return PlaneInfo(
                path,
                int(index),
                (int(header["NAXIS2"]), int(header["NAXIS1"])),
                dtype,
                header,
                comp,
                reason,
            )
    raise ValueError(
        f"No 2D image HDU found in {path}"
        if hdu is None
        else f"FITS HDU {hdu} of {path} is not a 2D image"
    )


def inspect_plane(path: str | Path, hdu: int | None = None) -> PlaneInfo:
    """Header-only description of one image HDU (the first 2D image when
    ``hdu`` is None).
    """
    resolved = Path(path).expanduser().resolve()
    if _fits_io is None:
        return _inspect_fallback(resolved, hdu)
    try:
        info = _fits_io.inspect_fits_image(resolved, hdu=hdu)
    except ValueError as exc:
        raise ValueError(f"{resolved}: {exc}") from exc
    return PlaneInfo(
        info.path,
        info.hdu,
        info.shape,
        info.dtype,
        info.header,
        info.compression,
        info.xdr_unsupported_reason,
    )


def hdu_index(path: str | Path, selector: int | str | None) -> int:
    """Resolve an HDU given as an index, an EXTNAME, or None (first 2D
    image).
    """
    if selector is None or isinstance(selector, int):
        return inspect_plane(path, selector).hdu
    from astropy.io import fits

    with fits.open(
        Path(path).expanduser().resolve(), memmap=False, lazy_load_hdus=True
    ) as hdus:
        for index, h in enumerate(hdus):
            if (
                str(h.header.get("EXTNAME", "")).strip().upper()
                == str(selector).strip().upper()
            ):
                return index
    raise ValueError(f"{path}: no HDU named {selector!r}")


class PlaneReader:
    """Reads image planes to host or device arrays under one reader policy.

    ``device=True`` returns CuPy arrays on ``stream`` (the current stream when
    None); ``device=False``
    returns NumPy arrays and is what the host tests use.
    """

    def __init__(
        self, reader: str = "auto", *, device: bool = True, stream=None
    ) -> None:
        if reader not in ("auto", "xdr", "astropy"):
            raise ValueError("reader must be auto, xdr, or astropy")
        self.reader = reader
        self.device = device
        self.stream = stream
        self.receipts: list[dict[str, Any]] = []

    @property
    def xp(self):
        if self.device:
            import cupy

            return cupy
        return np

    def read(
        self, path: str | Path, hdus: Sequence[int]
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Read the listed HDUs of one file; returns the arrays and the read
        receipt.
        """
        resolved = Path(path).expanduser().resolve()
        if _fits_io is not None:
            try:
                result = _fits_io.read_fits_images(
                    resolved,
                    list(hdus),
                    reader=self.reader,
                    device=self.device,
                    stream=self.stream,
                )
            except ValueError as exc:
                raise ValueError(f"{resolved}: {exc}") from exc
            receipt = {"path": str(resolved), **result.metadata()}
            self.receipts.append(receipt)
            return tuple(result.arrays), receipt
        return self._read_fallback(resolved, tuple(hdus))

    def read_batch(
        self, paths: Sequence[str | Path], hdus: Sequence[int]
    ) -> tuple[list[tuple[Any, ...]], list[dict[str, Any]]]:
        """Read the same HDUs from several files.

        Takes the one-call xDR batch path when every file is xDR-eligible, all
        share shapes and
        dtypes per HDU, and the policy allows xDR; otherwise reads file by
        file under the policy.
        """
        paths = [Path(p).expanduser().resolve() for p in paths]
        hdus = tuple(int(h) for h in hdus)
        if (
            self.device
            and self.reader != "astropy"
            and len(paths) > 1
            and _xdr_available()
        ):
            infos = [[inspect_plane(p, h) for h in hdus] for p in paths]
            eligible = all(
                i.xdr_unsupported_reason is None for row in infos for i in row
            )
            same = all(
                len({(row[k].shape, row[k].dtype.str) for row in infos}) == 1
                for k in range(len(hdus))
            )
            if eligible and same:
                from cuphoton.xdr import batch_to_device

                stacks = batch_to_device(
                    [str(p) for p in paths],
                    hdu_indices=hdus,
                    stream=self.stream,
                )
                receipts = []
                for n, p in enumerate(paths):
                    receipts.append(
                        {
                            "path": str(p),
                            "reader": "xdr",
                            "requested_reader": self.reader,
                            "location": "device",
                            "fallback_reason": None,
                            "batched": True,
                            "hdus": [
                                {
                                    "hdu": h,
                                    "shape": list(stacks[k][n].shape),
                                    "dtype": stacks[k][n].dtype.str,
                                    "compression": infos[n][k].compression,
                                }
                                for k, h in enumerate(hdus)
                            ],
                            "decoded_bytes": int(
                                sum(
                                    stacks[k][n].nbytes
                                    for k in range(len(hdus))
                                )
                            ),
                        }
                    )
                self.receipts.extend(receipts)
                return [
                    tuple(stacks[k][n] for k in range(len(hdus)))
                    for n in range(len(paths))
                ], receipts
        out, receipts = [], []
        for p in paths:
            arrays, receipt = self.read(p, hdus)
            out.append(arrays)
            receipts.append(receipt)
        return out, receipts

    def _read_fallback(self, path: Path, hdus: tuple[int, ...]):
        from astropy.io import fits

        infos = [inspect_plane(path, h) for h in hdus]
        reason = next(
            (
                i.xdr_unsupported_reason
                for i in infos
                if i.xdr_unsupported_reason
            ),
            None,
        )
        use_xdr = (
            self.device
            and self.reader != "astropy"
            and reason is None
            and _xdr_available()
        )
        if self.reader == "xdr" and not use_xdr:
            raise ValueError(
                f"{path}: cannot use xDR: "
                f"{reason or 'GPU dependencies unavailable'}"
            )
        if use_xdr:
            from cuphoton.xdr import batch_to_device

            stacks = batch_to_device(
                [str(path)], hdu_indices=hdus, stream=self.stream
            )
            arrays = tuple(s[0] for s in stacks)
            actual, fallback = "xdr", None
        else:
            with fits.open(path, memmap=False) as source:
                arrays = tuple(
                    np.array(
                        np.asarray(source[h].data),
                        dtype=np.asarray(source[h].data).dtype.newbyteorder(
                            "="
                        ),
                        copy=True,
                    )
                    for h in hdus
                )
            if self.device:
                import cupy as cp

                stream = (
                    self.stream
                    if self.stream is not None
                    else cp.cuda.get_current_stream()
                )
                with stream:
                    arrays = tuple(
                        cp.asarray(a, blocking=True) for a in arrays
                    )
                stream.synchronize()
            actual, fallback = (
                "astropy",
                (reason if self.reader != "astropy" else None),
            )
        receipt = {
            "path": str(path),
            "reader": actual,
            "requested_reader": self.reader,
            "location": "device" if self.device else "host",
            "fallback_reason": fallback,
            "hdus": [
                {
                    "hdu": h,
                    "shape": list(a.shape),
                    "dtype": a.dtype.str,
                    "compression": i.compression,
                }
                for h, a, i in zip(hdus, arrays, infos, strict=True)
            ],
            "decoded_bytes": int(sum(a.nbytes for a in arrays)),
        }
        self.receipts.append(receipt)
        return arrays, receipt
