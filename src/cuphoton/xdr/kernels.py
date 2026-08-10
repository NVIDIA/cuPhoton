# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CuPy kernels used by the GPU FITS reader.

Kernels are created lazily so that `import cuphoton.xdr` does not
require cupy. Each accessor caches the compiled kernel after first use.
"""

from __future__ import annotations

import functools

import numpy as np


@functools.lru_cache(maxsize=None)
def _byteswap_kernel(itemsize: int):
    """Return an in-place byteswap kernel for an item size.

    Operates on a uint8 device buffer viewed as groups of `itemsize` bytes;
    reverses each group.
    """
    import cupy as cp

    if itemsize == 2:
        return cp.ElementwiseKernel(
            "",
            "raw uint8 x",
            """
            long long base = (long long)i * 2;
            unsigned char b0 = x[base];
            x[base]     = x[base + 1];
            x[base + 1] = b0;
            """,
            name="fits_byteswap2",
        )
    if itemsize == 4:
        return cp.ElementwiseKernel(
            "",
            "raw uint8 x",
            """
            long long base = (long long)i * 4;
            unsigned char b0 = x[base];
            unsigned char b1 = x[base + 1];
            x[base]     = x[base + 3];
            x[base + 1] = x[base + 2];
            x[base + 2] = b1;
            x[base + 3] = b0;
            """,
            name="fits_byteswap4",
        )
    if itemsize == 8:
        return cp.ElementwiseKernel(
            "",
            "raw uint8 x",
            """
            long long base = (long long)i * 8;
            unsigned char b0 = x[base];
            unsigned char b1 = x[base + 1];
            unsigned char b2 = x[base + 2];
            unsigned char b3 = x[base + 3];
            x[base]     = x[base + 7];
            x[base + 1] = x[base + 6];
            x[base + 2] = x[base + 5];
            x[base + 3] = x[base + 4];
            x[base + 4] = b3;
            x[base + 5] = b2;
            x[base + 6] = b1;
            x[base + 7] = b0;
            """,
            name="fits_byteswap8",
        )
    raise ValueError(f"unsupported itemsize for byteswap: {itemsize}")


def byteswap_inplace(d_buf, itemsize: int) -> None:
    """Byteswap a device buffer in place. No-op for itemsize == 1."""
    if itemsize == 1:
        return
    import cupy as cp

    u8 = d_buf.view(cp.uint8).ravel()
    n_items = u8.size // itemsize
    if n_items * itemsize != u8.size:
        raise ValueError(
            f"buffer length {u8.size} not a multiple of itemsize {itemsize}"
        )
    _byteswap_kernel(itemsize)(u8, size=n_items)


def scatter_tiles_2d(
    d_tiles_concat,
    d_out,
    tile_byte_offsets,
    tile_origins_row,
    tile_origins_col,
    tile_heights,
    tile_widths,
    tile_src_off_row,
    tile_src_off_col,
    tile_full_widths,
    itemsize: int,
) -> None:
    """Copy decompressed tile bytes into a 2D output image on the device.

    Supports partial-tile copies for section (ROI) reads: each tile copies the
    sub-region ``(src_off_row..src_off_row+h, src_off_col..src_off_col+w)``
    from its decompressed block (laid out as ``tile_full_width`` pixels per
    row) into ``(orig_row..orig_row+h, orig_col..orig_col+w)`` in the output.

    For a full-image read pass ``src_off_*`` zeros and ``tile_full_widths``
    equal to the crop widths.
    """

    pix_stride_out = d_out.strides[0]  # row stride in bytes
    assert d_out.strides[1] == itemsize, "d_out must be C-contiguous"
    n_tiles = tile_byte_offsets.size

    kern = _scatter_tiles_2d_kernel(itemsize)
    threads_per_block = 256
    kern(
        (n_tiles,),
        (threads_per_block,),
        (
            d_tiles_concat,
            d_out,
            tile_byte_offsets,
            tile_origins_row,
            tile_origins_col,
            tile_heights,
            tile_widths,
            tile_src_off_row,
            tile_src_off_col,
            tile_full_widths,
            np.int64(pix_stride_out),
        ),
    )


@functools.lru_cache(maxsize=None)
def _scatter_tiles_2d_kernel(itemsize: int):
    import cupy as cp

    src = rf"""
    extern "C" __global__
    void scatter_tiles_{itemsize}(
        const unsigned char* __restrict__ tiles,
        unsigned char* __restrict__ out,
        const long long* __restrict__ tile_byte_offsets,
        const int* __restrict__ origins_row,
        const int* __restrict__ origins_col,
        const int* __restrict__ tile_h,
        const int* __restrict__ tile_w,
        const int* __restrict__ src_off_row,
        const int* __restrict__ src_off_col,
        const int* __restrict__ tile_full_w,
        long long out_row_stride_bytes
    ) {{
        const int t = blockIdx.x;
        const int ITEMSIZE = {itemsize};
        const long long src_off = tile_byte_offsets[t];
        const int r0 = origins_row[t];
        const int c0 = origins_col[t];
        const int h  = tile_h[t];
        const int w  = tile_w[t];
        const int sr = src_off_row[t];
        const int sc = src_off_col[t];
        const int fw = tile_full_w[t];
        const int total = h * w;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {{
            int r = i / w;
            int c = i - r * w;
            int src_linear = (sr + r) * fw + (sc + c);
            const unsigned char* src_pix =
                tiles + src_off + (long long)src_linear * ITEMSIZE;
            unsigned char* dst_pix =
                out + (long long)(r0 + r) * out_row_stride_bytes
                + (long long)(c0 + c) * ITEMSIZE;
            #pragma unroll
            for (int b = 0; b < ITEMSIZE; ++b) dst_pix[b] = src_pix[b];
        }}
    }}
    """
    return cp.RawKernel(src, f"scatter_tiles_{itemsize}")


@functools.lru_cache(maxsize=None)
def _unshuffle_gzip2_kernel(itemsize: int):
    import cupy as cp

    src = rf"""
    extern "C" __global__
    void unshuffle_gzip2_{itemsize}(
        const unsigned char* __restrict__ shuffled,
        unsigned char* __restrict__ interleaved,
        const long long* __restrict__ tile_byte_offsets,
        const long long* __restrict__ tile_byte_lengths
    ) {{
        const int t = blockIdx.x;
        const int ITEMSIZE = {itemsize};
        const long long off = tile_byte_offsets[t];
        const long long total = tile_byte_lengths[t];
        const long long n_pixels = total / ITEMSIZE;
        // Shuffled layout:
        // [plane_0 (n_pixels bytes), plane_1, ..., plane_(ITEMSIZE-1)]
        // Interleaved layout: [pix_0 (ITEMSIZE bytes), pix_1, ...]
        for (long long p = threadIdx.x; p < n_pixels; p += blockDim.x) {{
            for (int b = 0; b < ITEMSIZE; ++b) {{
                interleaved[off + p * ITEMSIZE + b] =
                    shuffled[off + (long long)b * n_pixels + p];
            }}
        }}
    }}
    """
    return cp.RawKernel(src, f"unshuffle_gzip2_{itemsize}")


def unshuffle_gzip2_tiles(
    d_shuffled,
    d_interleaved,
    tile_byte_offsets,
    tile_byte_lengths,
    itemsize: int,
):
    """Reorder GZIP_2 tile bytes from plane-major to pixel-major.

    `d_shuffled` and `d_interleaved` must be the same size; both are uint8.
    Each tile is processed independently (one CUDA block per tile).
    """
    if itemsize == 1:
        # Nothing to unshuffle; in-place no-op via alias.
        return
    kern = _unshuffle_gzip2_kernel(itemsize)
    n_tiles = tile_byte_offsets.size
    kern(
        (n_tiles,),
        (256,),
        (
            d_shuffled,
            d_interleaved,
            tile_byte_offsets,
            tile_byte_lengths,
        ),
    )


@functools.lru_cache(maxsize=None)
def _dequantize_int_to_float_kernel(int_itemsize: int, float_dtype_char: str):
    """Per-tile `out[p] = (float)int_buf[p] * zscale[t] + zzero[t]`.

    FITS ZSCALE/ZZERO quantization stores integer values in the compressed
    stream even when ZBITPIX is a float code, so dequantization must read
    integers (int32 or int64) and produce floats.

    Assumes NO_DITHER (or NONE); other dither methods must be handled on CPU.
    """
    import cupy as cp

    if int_itemsize == 4:
        int_type = "int"
    elif int_itemsize == 8:
        int_type = "long long"
    else:
        raise ValueError(
            "dequantize supports int32/int64 only, got itemsize "
            f"{int_itemsize}"
        )
    if float_dtype_char == "f4":
        float_type = "float"
    elif float_dtype_char == "f8":
        float_type = "double"
    else:
        raise ValueError(
            f"dequantize float output must be f4/f8, got {float_dtype_char}"
        )

    src = rf"""
    extern "C" __global__
    void dequantize_tiles_i{int_itemsize}_{float_dtype_char}(
        const {int_type}* __restrict__ int_buf,
        {float_type}* __restrict__ out,
        const int* __restrict__ origins_row,
        const int* __restrict__ origins_col,
        const int* __restrict__ tile_h,
        const int* __restrict__ tile_w,
        const double* __restrict__ zscale,
        const double* __restrict__ zzero,
        long long row_stride
    ) {{
        const int t = blockIdx.x;
        const int r0 = origins_row[t];
        const int c0 = origins_col[t];
        const int h  = tile_h[t];
        const int w  = tile_w[t];
        const {float_type} s = ({float_type})zscale[t];
        const {float_type} z = ({float_type})zzero[t];
        const int total = h * w;
        for (int i = threadIdx.x; i < total; i += blockDim.x) {{
            int r = i / w;
            int c = i - r * w;
            long long pos =
                (long long)(r0 + r) * row_stride + (long long)(c0 + c);
            out[pos] = ({float_type})int_buf[pos] * s + z;
        }}
    }}
    """
    name = f"dequantize_tiles_i{int_itemsize}_{float_dtype_char}"
    return cp.RawKernel(src, name)


def dequantize_int_to_float(
    d_int,
    d_float_out,
    origins_row,
    origins_col,
    tile_h,
    tile_w,
    zscale,
    zzero,
):
    """Read quantized ints, dequantize per tile, and write floats.

    Both arrays must be 2D C-contiguous with the same shape. `zscale` and
    `zzero` are float64 cupy arrays with one entry per tile.
    """
    import cupy as cp

    if d_int.shape != d_float_out.shape:
        raise ValueError("d_int and d_float_out must have the same shape")
    if d_int.dtype not in (cp.int32, cp.int64):
        raise ValueError("d_int must be int32 or int64")
    float_char = "f4" if d_float_out.dtype == cp.float32 else "f8"
    kern = _dequantize_int_to_float_kernel(d_int.itemsize, float_char)
    n_tiles = origins_row.size
    row_stride = d_float_out.strides[0] // d_float_out.itemsize
    kern(
        (n_tiles,),
        (256,),
        (
            d_int,
            d_float_out,
            origins_row,
            origins_col,
            tile_h,
            tile_w,
            zscale,
            zzero,
            np.int64(row_stride),
        ),
    )


def apply_bzero_bscale(d_arr, bzero: float, bscale: float):
    """Apply ``y = bscale * x + bzero`` in place."""
    if bzero == 0.0 and bscale == 1.0:
        return d_arr
    # Let cupy broadcast — it will fuse into a single kernel launch.
    if bscale != 1.0:
        d_arr *= bscale
    if bzero != 0.0:
        d_arr += bzero
    return d_arr
