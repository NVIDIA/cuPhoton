# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""GPU-native readers for FITS image HDUs.

- `GpuImageReader`: uncompressed `ImageHDU` / `PrimaryHDU` pixel data.
  One `kvikio.CuFile.pread` straight into a cupy buffer, then a device-side
  big->little endian byteswap if needed.

- `GpuCompImageReader`: compressed GZIP_1 / GZIP_2 `CompImageHDU`.
"""

from __future__ import annotations

import numpy as np

from .gds import GdsHeapLoader, pread_to_device
from .kernels import (
    apply_bzero_bscale,
    byteswap_inplace,
    dequantize_int_to_float,
    scatter_tiles_2d,
    unshuffle_gzip2_tiles,
)
from .nvcomp_batch import (
    _native_device_empty,
    _native_device_empty_uint8,
    gpu_gzip_decompress_batch,
)

# FITS BITPIX -> (numpy dtype char, itemsize). On disk FITS is big-endian.
_BITPIX_TO_DTYPE = {
    8: ("u1", 1),
    16: ("i2", 2),
    32: ("i4", 4),
    64: ("i8", 8),
    -32: ("f4", 4),
    -64: ("f8", 8),
}


def _hdu_path(hdu) -> str:
    f = getattr(hdu, "_file", None)
    if f is None or not hasattr(f, "name"):
        raise RuntimeError(
            "ImageHDU has no backing file; GPU reader requires "
            "fits.open()'d HDU"
        )
    return f.name


def _image_shape(header) -> tuple[int, ...]:
    naxis = int(header["NAXIS"])
    # FITS stores axes in FORTRAN order; numpy wants C order.
    return tuple(int(header[f"NAXIS{i}"]) for i in range(naxis, 0, -1))


class GpuImageReader:
    """Read an uncompressed ImageHDU directly into a cupy.ndarray via GDS."""

    def __init__(self, hdu):
        if int(hdu.header.get("NAXIS", 0)) == 0:
            raise ValueError("HDU has no data (NAXIS=0)")
        bitpix = int(hdu.header["BITPIX"])
        if bitpix not in _BITPIX_TO_DTYPE:
            raise ValueError(f"unsupported BITPIX {bitpix}")
        self.hdu = hdu
        self.path = _hdu_path(hdu)
        self.file_offset = int(hdu._data_offset)
        self.shape = _image_shape(hdu.header)
        self.dtype_char, self.itemsize = _BITPIX_TO_DTYPE[bitpix]
        self.nbytes = int(np.prod(self.shape)) * self.itemsize
        self.bzero = float(hdu.header.get("BZERO", 0.0))
        self.bscale = float(hdu.header.get("BSCALE", 1.0))

    def read(self, *, out=None, stream=None, loader=None):
        """Read pixel data into a `cupy.ndarray` via kvikio.

        `loader` is accepted for API parity with `GpuCompImageReader`; the
        uncompressed path is a single pread so reusing a CuFile is still a
        small win when loading many HDUs from one file.
        """
        import cupy as cp

        native_dtype = cp.dtype(self.dtype_char)
        if out is None:
            out = cp.empty(self.shape, dtype=native_dtype)
        else:
            if out.shape != self.shape or out.dtype != native_dtype:
                raise ValueError(
                    f"out buffer shape/dtype mismatch: expected "
                    f"{self.shape} {native_dtype}, got "
                    f"{out.shape} {out.dtype}"
                )

        with stream or cp.cuda.Stream.null:
            if loader is not None:
                fut = loader._file().pread(
                    out,
                    size=self.nbytes,
                    file_offset=self.file_offset,
                )
                fut.get()
            else:
                pread_to_device(self.path, self.file_offset, self.nbytes, out)
            # FITS on disk is big-endian; byteswap to native little-endian.
            byteswap_inplace(out, self.itemsize)
            if self.bzero != 0.0 or self.bscale != 1.0:
                # BZERO/BSCALE may be integer-coded; promote for scaling.
                if out.dtype.kind in ("i", "u"):
                    out = out.astype(
                        cp.float32 if self.itemsize <= 4 else cp.float64
                    )
                apply_bzero_bscale(out, self.bzero, self.bscale)

        return out


def _tile_shape_from_header(header) -> tuple[int, ...]:
    naxis = int(header["ZNAXIS"])
    return tuple(int(header[f"ZTILE{i}"]) for i in range(naxis, 0, -1))


def _data_shape_from_header(header) -> tuple[int, ...]:
    naxis = int(header["ZNAXIS"])
    return tuple(int(header[f"ZNAXIS{i}"]) for i in range(naxis, 0, -1))


def _validate_comp_geometry(
    data_shape: tuple[int, ...], tile_shape: tuple[int, ...]
) -> None:
    if any(dim <= 0 for dim in data_shape) or any(
        dim <= 0 for dim in tile_shape
    ):
        raise ValueError(
            "invalid compressed image geometry: "
            f"ZNAXIS shape {data_shape}, ZTILE shape {tile_shape}; "
            "all dimensions must be positive"
        )


class GpuCompImageReader:
    """Read a GZIP_1 (M3) / GZIP_2 (M4) CompImageHDU into a cupy.ndarray.

    Pipeline: GDS heap read -> nvCOMP batched DEFLATE -> device-side byteswap
    -> (GZIP_2 unshuffle, M4) -> scatter into output image.
    """

    SUPPORTED_GZIP = frozenset({"GZIP_1", "GZIP_2"})

    def __init__(self, hdu):
        self.hdu = hdu
        self.bt = hdu._bintable
        self.path = self.bt._file.name
        self.compression_type = hdu.compression_type
        if self.compression_type not in self.SUPPORTED_GZIP:
            raise NotImplementedError(
                "GpuCompImageReader supports GZIP_1/GZIP_2 only; got "
                f"{self.compression_type!r}"
            )

        zhdr = self.bt.header
        self.quantized = "ZSCALE" in self.bt.data.dtype.names
        if self.quantized:
            zquantiz = (zhdr.get("ZQUANTIZ") or "NO_DITHER").upper()
            if zquantiz not in ("NO_DITHER", "NONE"):
                raise NotImplementedError(
                    f"ZQUANTIZ={zquantiz!r} (dithered) not supported on "
                    "GPU path"
                )
            self.zscale = np.array(self.bt.data["ZSCALE"], dtype=np.float64)
            self.zzero = np.array(self.bt.data["ZZERO"], dtype=np.float64)
        else:
            self.zscale = None
            self.zzero = None
        self.data_shape = _data_shape_from_header(zhdr)
        self.tile_shape = _tile_shape_from_header(zhdr)
        _validate_comp_geometry(self.data_shape, self.tile_shape)
        self.zbitpix = int(zhdr["ZBITPIX"])
        if self.zbitpix not in _BITPIX_TO_DTYPE:
            raise ValueError(f"unsupported ZBITPIX {self.zbitpix}")
        self.dtype_char, self.itemsize = _BITPIX_TO_DTYPE[self.zbitpix]

        comp_col = np.array(
            self.bt.data["COMPRESSED_DATA"]
        )  # shape (N, 2): (size, offset)
        if comp_col.ndim != 2 or comp_col.shape[1] != 2:
            raise ValueError(
                f"unexpected COMPRESSED_DATA shape {comp_col.shape}"
            )
        self.lengths = comp_col[:, 0].astype(np.int64)
        heap_base = int(self.bt._data_offset) + int(self.bt._theap)
        self.abs_offsets = heap_base + comp_col[:, 1].astype(np.int64)
        self.n_tiles = self.lengths.size

    def _tile_grid_2d(self, section: tuple | None = None):
        """Enumerate 2D tiles, optionally cropped to a section rectangle.

        Parameters
        ----------
        section : tuple of slice or None
            (row_slice, col_slice). When provided, only tiles overlapping the
            rectangle are returned and their origins are shifted so that
            ``origins_row/col`` address pixels in the ROI output (which has
            shape ``(row_slice.stop-row_slice.start,
            col_slice.stop-col_slice.start)``).

        Returns:
          tile_indices (np.int64[n])  — COMPRESSED_DATA row for each tile
          origins_row  (np.int32[n])  — top-left row inside OUTPUT
          origins_col  (np.int32[n])  — top-left col inside OUTPUT
          heights      (np.int32[n])  — crop height of tile inside the ROI
          widths       (np.int32[n])  — crop width of tile inside the ROI
          out_bytes    (np.int64[n])  — full-tile decompressed byte size
          src_offset_r (np.int32[n])  — row offset inside decompressed tile
          src_offset_c (np.int32[n])  — col offset inside decompressed tile
          out_shape    (tuple)         — shape of the output buffer
        """
        if len(self.data_shape) != 2:
            raise NotImplementedError("only 2D images supported")

        H, W = self.data_shape
        th, tw = self.tile_shape

        if section is None:
            roi_r0, roi_r1 = 0, H
            roi_c0, roi_c1 = 0, W
        else:
            sr, sc = section
            roi_r0 = 0 if sr.start is None else int(sr.start)
            roi_r1 = H if sr.stop is None else int(sr.stop)
            roi_c0 = 0 if sc.start is None else int(sc.start)
            roi_c1 = W if sc.stop is None else int(sc.stop)
            in_bounds = (
                0 <= roi_r0 < roi_r1 <= H and 0 <= roi_c0 < roi_c1 <= W
            )
            if not in_bounds:
                raise ValueError(
                    f"section out of bounds: {section} for shape "
                    f"{self.data_shape}"
                )
            if sr.step not in (None, 1) or sc.step not in (None, 1):
                raise NotImplementedError(
                    "strided section slices not supported"
                )

        out_h = roi_r1 - roi_r0
        out_w = roi_c1 - roi_c0

        n_cols_in_grid = (W + tw - 1) // tw

        idx = []
        orig_r = []
        orig_c = []
        heights = []
        widths = []
        out_bytes = []
        src_off_r = []
        src_off_c = []

        for tr0 in range(0, H, th):
            t_h_full = min(th, H - tr0)
            for tc0 in range(0, W, tw):
                t_w_full = min(tw, W - tc0)
                # Tile covers rows [tr0, tr0+t_h_full) and columns
                # [tc0, tc0+t_w_full).
                inter_r0 = max(tr0, roi_r0)
                inter_r1 = min(tr0 + t_h_full, roi_r1)
                inter_c0 = max(tc0, roi_c0)
                inter_c1 = min(tc0 + t_w_full, roi_c1)
                if inter_r0 >= inter_r1 or inter_c0 >= inter_c1:
                    continue
                row_i = (tr0 // th) * n_cols_in_grid + (tc0 // tw)
                idx.append(row_i)
                # Origin in OUTPUT buffer space.
                orig_r.append(inter_r0 - roi_r0)
                orig_c.append(inter_c0 - roi_c0)
                heights.append(inter_r1 - inter_r0)
                widths.append(inter_c1 - inter_c0)
                out_bytes.append(t_h_full * t_w_full * self.itemsize)
                # Offset inside decompressed tile for partial-tile copies.
                src_off_r.append(inter_r0 - tr0)
                src_off_c.append(inter_c0 - tc0)
                # Also need the full tile width for scatter indexing.
        return (
            np.asarray(idx, dtype=np.int64),
            np.asarray(orig_r, dtype=np.int32),
            np.asarray(orig_c, dtype=np.int32),
            np.asarray(heights, dtype=np.int32),
            np.asarray(widths, dtype=np.int32),
            np.asarray(out_bytes, dtype=np.int64),
            np.asarray(src_off_r, dtype=np.int32),
            np.asarray(src_off_c, dtype=np.int32),
            (out_h, out_w),
        )

    # ------------------------------------------------------------------ #
    # Read path is split into three reusable pieces so the multi-file
    # prefetch pipeline (`prefetch.py`) can compute plans on a background
    # thread, stage the compressed heap to device, and call the decode
    # kernels without duplicating this logic.
    # ------------------------------------------------------------------ #

    def prepare_plan(self, section=None) -> dict:
        """CPU-only planning: tile grid + selected offsets/lengths/etc.

        Returned dict is self-contained so it can be shipped across threads
        (numpy arrays, plain ints, one tuple). Safe to compute without any
        CUDA context.
        """
        (
            tile_idx,
            origins_r,
            origins_c,
            heights,
            widths,
            out_bytes,
            src_off_r,
            src_off_c,
            out_shape,
        ) = self._tile_grid_2d(section=section)

        sel_abs_offsets = self.abs_offsets[tile_idx]
        sel_lengths = self.lengths[tile_idx]

        H, W = self.data_shape
        th, tw = self.tile_shape
        n_cols_in_grid = (W + tw - 1) // tw
        tile_full_w = np.empty(tile_idx.size, dtype=np.int32)
        for k, row_i in enumerate(tile_idx):
            tc0 = (int(row_i) % n_cols_in_grid) * tw
            tile_full_w[k] = min(tw, W - tc0)

        plan = {
            "tile_idx": tile_idx,
            "sel_abs_offsets": sel_abs_offsets,
            "sel_lengths": sel_lengths,
            "origins_r": origins_r,
            "origins_c": origins_c,
            "heights": heights,
            "widths": widths,
            "out_bytes": out_bytes,
            "src_off_r": src_off_r,
            "src_off_c": src_off_c,
            "tile_full_w": tile_full_w,
            "out_shape": out_shape,
            "compression_type": self.compression_type,
            "itemsize": self.itemsize,
            "dtype_char": self.dtype_char,
            "zbitpix": self.zbitpix,
            "quantized": self.quantized,
        }
        if self.quantized:
            plan["sel_zscale"] = self.zscale[tile_idx]
            plan["sel_zzero"] = self.zzero[tile_idx]
        return plan

    @staticmethod
    def _alloc_output(plan: dict, out=None, *, use_native_pool: bool = False):
        """Validate or allocate output and scatter intermediate."""
        import cupy as cp

        native_dtype = cp.dtype(plan["dtype_char"])
        out_shape = plan["out_shape"]
        if out is None:
            out = cp.empty(out_shape, dtype=native_dtype)
        else:
            if out.shape != out_shape or out.dtype != native_dtype:
                raise ValueError(
                    f"out buffer shape/dtype mismatch: expected "
                    f"{out_shape} {native_dtype}, got {out.shape} {out.dtype}"
                )
        if not out.flags.c_contiguous:
            raise ValueError("out buffer must be C-contiguous")

        if plan["quantized"]:
            zbitpix = plan["zbitpix"]
            if zbitpix == -32:
                scatter_dtype, scatter_itemsize = cp.int32, 4
            elif zbitpix == -64:
                scatter_dtype, scatter_itemsize = cp.int64, 8
            else:
                raise ValueError(
                    "quantized images must have ZBITPIX -32 or -64, "
                    f"got {zbitpix}"
                )
            d_scatter_target = (
                _native_device_empty(out_shape, scatter_dtype)
                if use_native_pool
                else cp.empty(out_shape, dtype=scatter_dtype)
            )
        else:
            d_scatter_target = out
            scatter_itemsize = plan["itemsize"]
        return out, d_scatter_target, scatter_itemsize

    @staticmethod
    def inflate_device_heap(
        d_concat,
        rel_offsets,
        *,
        lengths,
        out_bytes,
        stream=None,
        keepalive=None,
    ):
        """Run batched nvCOMP inflate for gzip tiles resident on the device.

        `d_concat` is a uint8 cupy.ndarray holding the compressed tile bytes
        and `rel_offsets[i]` is the start offset of tile i inside `d_concat`.
        `lengths` and `out_bytes` are per-tile compressed and decompressed
        byte counts. Returns concatenated decompressed bytes plus per-tile
        offsets into that output buffer.
        """
        import cupy as cp

        with stream or cp.cuda.Stream.null:
            return gpu_gzip_decompress_batch(
                d_concat,
                rel_offsets,
                lengths,
                out_bytes,
                gzip_wrapped=True,
                use_native_pool=keepalive is not None,
                keepalive=keepalive,
            )

    @staticmethod
    def postprocess_decoded_tiles(
        d_pixels,
        tile_byte_offsets_np,
        plan: dict,
        *,
        out=None,
        stream=None,
        keepalive=None,
    ):
        """Scatter already-inflated tile bytes into the final image output."""
        import cupy as cp

        use_native_pool = keepalive is not None
        out, d_scatter_target, scatter_itemsize = (
            GpuCompImageReader._alloc_output(
                plan,
                out,
                use_native_pool=use_native_pool,
            )
        )
        if keepalive is not None:
            keepalive.append(d_scatter_target)

        with stream or cp.cuda.Stream.null:
            d_tile_offsets = cp.asarray(tile_byte_offsets_np)
            d_tile_byte_lengths = cp.asarray(plan["out_bytes"])
            if keepalive is not None:
                keepalive.extend([d_tile_offsets, d_tile_byte_lengths])

            # 1. GZIP_2: byte-plane unshuffle per tile.
            if plan["compression_type"] == "GZIP_2" and plan["itemsize"] > 1:
                d_inter = (
                    _native_device_empty_uint8(d_pixels.nbytes).reshape(
                        d_pixels.shape
                    )
                    if use_native_pool
                    else cp.empty_like(d_pixels)
                )
                unshuffle_gzip2_tiles(
                    d_pixels,
                    d_inter,
                    d_tile_offsets,
                    d_tile_byte_lengths,
                    plan["itemsize"],
                )
                d_pixels = d_inter
                if keepalive is not None:
                    keepalive.append(d_inter)

            # 2. Byteswap (FITS on-disk is big-endian, host is little-endian).
            if plan["itemsize"] > 1:
                byteswap_inplace(d_pixels, plan["itemsize"])

            # 3. Scatter tiles into int/float output.
            d_origins_r = cp.asarray(plan["origins_r"])
            d_origins_c = cp.asarray(plan["origins_c"])
            d_heights = cp.asarray(plan["heights"])
            d_widths = cp.asarray(plan["widths"])
            d_src_off_r = cp.asarray(plan["src_off_r"])
            d_src_off_c = cp.asarray(plan["src_off_c"])
            d_tile_full_w = cp.asarray(plan["tile_full_w"])
            if keepalive is not None:
                keepalive.extend(
                    [
                        d_origins_r,
                        d_origins_c,
                        d_heights,
                        d_widths,
                        d_src_off_r,
                        d_src_off_c,
                        d_tile_full_w,
                    ]
                )
            scatter_tiles_2d(
                d_pixels,
                d_scatter_target,
                d_tile_offsets,
                d_origins_r,
                d_origins_c,
                d_heights,
                d_widths,
                d_src_off_r,
                d_src_off_c,
                d_tile_full_w,
                scatter_itemsize,
            )

            # 4. Dequantize int -> float when ZSCALE/ZZERO are present.
            if plan["quantized"]:
                d_sel_zscale = cp.asarray(plan["sel_zscale"])
                d_sel_zzero = cp.asarray(plan["sel_zzero"])
                if keepalive is not None:
                    keepalive.extend([d_sel_zscale, d_sel_zzero])
                dequantize_int_to_float(
                    d_scatter_target,
                    out,
                    d_origins_r,
                    d_origins_c,
                    d_heights,
                    d_widths,
                    d_sel_zscale,
                    d_sel_zzero,
                )

        return out

    @staticmethod
    def decode_from_device_heap(
        d_concat,
        rel_offsets,
        plan: dict,
        *,
        out=None,
        stream=None,
        keepalive=None,
    ):
        """Run the decode pipeline on a pre-loaded compressed heap buffer.

        `d_concat` is a uint8 cupy.ndarray holding the compressed tile bytes
        of the tiles referenced in `plan["sel_*"]`; `rel_offsets[i]` is the
        start offset of tile i inside `d_concat`. This is what the prefetch
        consumer calls after staging its pinned host heap to device.
        """
        d_pixels, tile_byte_offsets_np = (
            GpuCompImageReader.inflate_device_heap(
                d_concat,
                rel_offsets,
                lengths=plan["sel_lengths"],
                out_bytes=plan["out_bytes"],
                stream=stream,
                keepalive=keepalive,
            )
        )
        if keepalive is not None:
            keepalive.extend([d_concat, d_pixels])
        return GpuCompImageReader.postprocess_decoded_tiles(
            d_pixels,
            tile_byte_offsets_np,
            plan,
            out=out,
            stream=stream,
            keepalive=keepalive,
        )

    def read(self, *, out=None, stream=None, section=None, loader=None):
        """Read and decode this HDU into a `cupy.ndarray`.

        Parameters
        ----------
        loader : GdsHeapLoader or None
            Reuse an existing persistent loader (avoids the ~22 ms
            kvikio `CompatModeManager` setup per HDU). If `None`, a fresh
            loader is created and closed within this call.
        """
        import cupy as cp

        plan = self.prepare_plan(section=section)

        owns_loader = loader is None
        if owns_loader:
            loader = GdsHeapLoader(self.path)

        keepalive = [] if stream is not None else None
        with stream or cp.cuda.Stream.null:
            handle = loader.load_tiles_async(
                plan["sel_abs_offsets"], plan["sel_lengths"]
            )
            d_concat, rel_offsets = handle.wait()
            out = GpuCompImageReader.decode_from_device_heap(
                d_concat,
                rel_offsets,
                plan,
                out=out,
                stream=stream,
                keepalive=keepalive,
            )
        if stream is not None:
            stream.synchronize()

        if owns_loader:
            loader.close()
        return out
