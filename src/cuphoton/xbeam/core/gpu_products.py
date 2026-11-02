# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Generic fused GPU accumulation and scatter operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..runtime.policy import current_policy
from .product_kernels import (
    _batched_visibility_kernel,
    _multi_visibility_kernel,
    _visibility_kernel,
)


@dataclass(frozen=True, slots=True)
class _PreparedIndexArray:
    """Host-validated indices paired with their asynchronous device copy."""

    host: np.ndarray
    device: Any

    @property
    def size(self) -> int:
        return int(self.host.size)


def _validate_visibility_ranges(
    group_starts: np.ndarray,
    group_lengths: np.ndarray,
    output_offsets: np.ndarray,
    weight_offsets: np.ndarray,
    *,
    block_count: int,
    output_channels: int,
    output_size: int,
    weight_size: int,
) -> None:
    starts = np.asarray(group_starts, dtype=np.int64)
    lengths = np.asarray(group_lengths, dtype=np.int64)
    if np.any(starts < 0) or np.any(lengths <= 0):
        raise ValueError(
            "visibility group ranges must be positive and in range"
        )
    if np.any(starts > block_count - lengths):
        raise ValueError("visibility group range exceeds the spectrum blocks")
    offsets = np.asarray(output_offsets, dtype=np.int64)
    if output_channels > output_size or np.any(offsets < 0):
        raise ValueError("visibility output offset is out of range")
    if np.any(offsets > output_size - output_channels):
        raise ValueError("visibility output offset is out of range")
    flat_offsets = offsets.reshape(-1)
    if flat_offsets.size > 1:
        ordered_offsets = np.sort(flat_offsets)
        if np.any(
            ordered_offsets[1:] < ordered_offsets[:-1] + output_channels
        ):
            raise ValueError("visibility output intervals overlap")
    weight_indices = np.asarray(weight_offsets, dtype=np.int64)
    if np.any(weight_indices < 0) or np.any(weight_indices >= weight_size):
        raise ValueError("visibility weight offset is out of range")
    if np.unique(weight_indices).size != weight_indices.size:
        raise ValueError("visibility weight offsets must be unique")


def _prepare_index_array(
    name: str,
    values: Any,
    shape: tuple[int, ...],
    device_dtype: Any,
    xp: Any,
) -> tuple[np.ndarray, Any | None]:
    if isinstance(values, _PreparedIndexArray):
        host = values.host
        device = values.device
        if (
            host.dtype != np.int64
            or host.shape != shape
            or not host.flags.c_contiguous
            or not isinstance(device, xp.ndarray)
            or device.dtype != device_dtype
            or device.shape != shape
            or not device.flags.c_contiguous
        ):
            raise TypeError(
                f"prepared {name} does not match the required shape or dtype"
            )
        return host, device
    if isinstance(values, xp.ndarray):
        if values.dtype != device_dtype or values.shape != shape:
            raise TypeError(
                f"{name} must contain signed {device_dtype} indices with "
                f"shape {shape}"
            )
        if not values.flags.c_contiguous:
            raise TypeError(f"{name} must be contiguous")
        return (
            np.ascontiguousarray(xp.asnumpy(values), dtype=np.int64),
            values,
        )
    host = np.asarray(values)
    if host.dtype.kind != "i" or host.shape != shape:
        raise TypeError(
            f"{name} must contain signed integer indices with shape {shape}"
        )
    return np.ascontiguousarray(host, dtype=np.int64), None


def prepare_visibility_index(
    values: Any,
    *,
    device_dtype: Any,
    xp: Any,
) -> _PreparedIndexArray:
    """Validate and upload host indices without a device readback."""

    host_values = np.asarray(values)
    if host_values.dtype.kind != "i":
        raise TypeError("visibility indices must be signed integers")
    host = np.ascontiguousarray(host_values, dtype=np.int64)
    return _PreparedIndexArray(host, xp.asarray(host, dtype=device_dtype))


def visibility_index_view(
    prepared: _PreparedIndexArray, index: Any
) -> _PreparedIndexArray:
    """Return a contiguous host/device view without another transfer."""

    host = prepared.host[index]
    device = prepared.device[index]
    if not host.flags.c_contiguous or not device.flags.c_contiguous:
        raise ValueError("prepared visibility index view must be contiguous")
    return _PreparedIndexArray(host, device)


def _device_index_array(
    host: np.ndarray,
    device: Any | None,
    dtype: Any,
    xp: Any,
) -> Any:
    return device if device is not None else xp.asarray(host, dtype=dtype)


def visibility_product(
    spectrum_a: Any,
    spectrum_b: Any,
    *,
    normalization: float,
    channels_to_average: int,
    weight_a: Any,
    weight_b: Any = None,
    xp: Any,
    output: Any = None,
    weight_sum: Any = None,
) -> tuple[Any, Any]:
    """Fuse visibility accumulation, fixed normalization, and weight sum."""

    supported = {xp.dtype(xp.complex64), xp.dtype(xp.complex128)}
    if (
        spectrum_a.dtype not in supported
        or spectrum_b.dtype != spectrum_a.dtype
    ):
        raise TypeError(
            "RawKernel visibility products require matching complex spectra"
        )
    if spectrum_a.shape != spectrum_b.shape or spectrum_a.ndim != 2:
        raise ValueError(
            "visibility spectra must have the same two-dimensional shape"
        )
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError(
            "visibility normalization must be finite and positive"
        )
    if channels_to_average <= 0:
        raise ValueError("channels_to_average must be positive")
    if not spectrum_a.flags.c_contiguous or not spectrum_b.flags.c_contiguous:
        raise TypeError("visibility spectra must be contiguous")
    block_count, input_channels = map(int, spectrum_a.shape)
    if not 0 < block_count <= np.iinfo(np.int32).max:
        raise ValueError("visibility block count is out of range")
    if not 0 < input_channels <= np.iinfo(np.int32).max:
        raise ValueError("visibility channel count is out of range")
    if input_channels % channels_to_average:
        raise ValueError("channel count is not divisible by averaging factor")
    output_channels = input_channels // channels_to_average
    if weight_a.dtype != xp.float64 or weight_a.shape != (block_count,):
        raise TypeError(
            "visibility weight_a must be one float64 value per block"
        )
    if not weight_a.flags.c_contiguous:
        raise TypeError("visibility weight_a must be contiguous")
    multiply_weights = weight_b is not None
    if weight_b is None:
        weight_b = weight_a
    elif weight_b.dtype != xp.float64 or weight_b.shape != (block_count,):
        raise TypeError(
            "visibility weight_b must be one float64 value per block"
        )
    if not weight_b.flags.c_contiguous:
        raise TypeError("visibility weight_b must be contiguous")
    if output is None:
        output = xp.empty(output_channels, dtype=spectrum_a.dtype)
    elif output.dtype != spectrum_a.dtype or output.shape != (
        output_channels,
    ):
        raise TypeError(
            "visibility output must match the spectrum dtype "
            "and output channel count"
        )
    elif not output.flags.c_contiguous:
        raise TypeError("visibility output must be contiguous")
    if weight_sum is None:
        weight_sum = xp.empty(1, dtype=xp.float64)
    elif weight_sum.dtype != xp.float64 or weight_sum.shape != (1,):
        raise TypeError(
            "visibility weight_sum output must contain one float64 value"
        )
    elif not weight_sum.flags.c_contiguous:
        raise TypeError("visibility weight_sum output must be contiguous")
    threads = 256
    averaging_mode = (
        channels_to_average if channels_to_average in (1, 4, 32) else 0
    )
    averaging_args: tuple[np.int32, ...]
    if averaging_mode:
        grid = ((input_channels + 31) // 32,)
        averaging_args = ()
    else:
        grid = ((output_channels + threads - 1) // threads,)
        averaging_args = (np.int32(channels_to_average),)
    _visibility_kernel(spectrum_a.dtype == xp.complex128, averaging_mode)(
        grid,
        (256,),
        (
            spectrum_a,
            spectrum_b,
            weight_a,
            weight_b,
            np.bool_(multiply_weights),
            np.int32(block_count),
            np.int32(input_channels),
            *averaging_args,
            np.float64(1.0 / normalization),
            output,
            weight_sum,
        ),
    )
    return output, weight_sum


def batched_visibility_product(
    spectrum_a: Any,
    spectrum_b: Any,
    *,
    normalization: float,
    channels_to_average: int,
    weight_a: Any,
    group_starts: Any,
    group_lengths: Any,
    output_offsets: Any,
    weight_offsets: Any,
    output: Any,
    weight_sum: Any,
    weight_b: Any = None,
    xp: Any,
    _validate_ranges: bool = True,
    accumulate: bool = False,
) -> None:
    """Accumulate many complete integration groups in one GPU launch."""

    if spectrum_a.dtype not in (xp.complex64, xp.complex128) or (
        spectrum_b.dtype != spectrum_a.dtype
    ):
        raise TypeError(
            "batched visibility spectra must have matching complex dtypes"
        )
    if spectrum_a.shape != spectrum_b.shape or spectrum_a.ndim != 2:
        raise ValueError(
            "batched visibility spectra must have matching matrices"
        )
    element_bytes = int(spectrum_a.dtype.itemsize)
    for name, spectrum in (
        ("spectrum_a", spectrum_a),
        ("spectrum_b", spectrum_b),
    ):
        if (
            int(spectrum.strides[1]) != element_bytes
            or int(spectrum.strides[0])
            < int(spectrum.shape[1]) * element_bytes
            or int(spectrum.strides[0]) % element_bytes
        ):
            raise TypeError(
                f"batched visibility {name} must have contiguous channels "
                "and a positive integral row stride"
            )
    spectrum_a_stride = int(spectrum_a.strides[0]) // element_bytes
    spectrum_b_stride = int(spectrum_b.strides[0]) // element_bytes
    if not np.isfinite(normalization) or normalization <= 0:
        raise ValueError(
            "visibility normalization must be finite and positive"
        )
    if channels_to_average <= 0:
        raise ValueError("channels_to_average must be positive")
    block_count, input_channels = map(int, spectrum_a.shape)
    if not 0 < block_count <= np.iinfo(np.int32).max:
        raise ValueError("batched visibility block count is out of range")
    if not 0 < input_channels <= np.iinfo(np.int32).max:
        raise ValueError("batched visibility channel count is out of range")
    if input_channels % channels_to_average:
        raise ValueError("channel count is not divisible by averaging factor")
    output_channels = input_channels // channels_to_average
    if weight_a.dtype != xp.float64 or weight_a.shape != (block_count,):
        raise TypeError(
            "batched visibility weight_a must match the block count"
        )
    if not weight_a.flags.c_contiguous:
        raise TypeError("batched visibility weight_a must be contiguous")
    multiply_weights = weight_b is not None
    if weight_b is None:
        weight_b = weight_a
    elif weight_b.dtype != xp.float64 or weight_b.shape != (block_count,):
        raise TypeError(
            "batched visibility weight_b must match the block count"
        )
    if not weight_b.flags.c_contiguous:
        raise TypeError("batched visibility weight_b must be contiguous")
    group_count = int(
        group_starts.size
        if hasattr(group_starts, "size")
        else np.asarray(group_starts).size
    )
    if group_count <= 0 or group_count > 65_535:
        raise ValueError("batched visibility requires 1..65535 groups")
    prepared_indices = {
        name: _prepare_index_array(name, values, (group_count,), dtype, xp)
        for name, values, dtype in (
            ("group_starts", group_starts, xp.int32),
            ("group_lengths", group_lengths, xp.int32),
            ("output_offsets", output_offsets, xp.int64),
            ("weight_offsets", weight_offsets, xp.int64),
        )
    }
    if output.dtype != spectrum_a.dtype or output.ndim != 1:
        raise TypeError(
            "batched visibility output must be a vector matching the spectra"
        )
    if weight_sum.dtype != xp.float64 or weight_sum.ndim != 1:
        raise TypeError(
            "batched visibility weight output must be a float64 vector"
        )
    if not output.flags.c_contiguous or not weight_sum.flags.c_contiguous:
        raise TypeError("batched visibility outputs must be contiguous")
    if _validate_ranges:
        _validate_visibility_ranges(
            prepared_indices["group_starts"][0],
            prepared_indices["group_lengths"][0],
            prepared_indices["output_offsets"][0],
            prepared_indices["weight_offsets"][0],
            block_count=block_count,
            output_channels=output_channels,
            output_size=int(output.size),
            weight_size=int(weight_sum.size),
        )
    group_starts = _device_index_array(
        *prepared_indices["group_starts"], xp.int32, xp
    )
    group_lengths = _device_index_array(
        *prepared_indices["group_lengths"], xp.int32, xp
    )
    output_offsets = _device_index_array(
        *prepared_indices["output_offsets"], xp.int64, xp
    )
    weight_offsets = _device_index_array(
        *prepared_indices["weight_offsets"], xp.int64, xp
    )

    threads = 256
    averaging_mode = (
        channels_to_average if channels_to_average in (1, 4, 32) else 0
    )
    averaging_args: tuple[np.int32, ...]
    if averaging_mode:
        channel_tiles = (input_channels + 31) // 32
        averaging_args = ()
    else:
        channel_tiles = (output_channels + threads - 1) // threads
        averaging_args = (np.int32(channels_to_average),)
    _batched_visibility_kernel(
        averaging_mode, spectrum_a.dtype == xp.complex128, accumulate
    )(
        (channel_tiles, group_count),
        (threads,),
        (
            spectrum_a,
            spectrum_b,
            np.int64(spectrum_a_stride),
            np.int64(spectrum_b_stride),
            weight_a,
            weight_b,
            np.bool_(multiply_weights),
            group_starts,
            group_lengths,
            output_offsets,
            weight_offsets,
            np.int32(input_channels),
            *averaging_args,
            np.float64(1.0 / normalization),
            output,
            weight_sum,
        ),
    )


def accumulate_grouped_cross_power(
    spectrum_a: Any,
    spectrum_b: Any,
    output: Any,
    pair_index: int,
    *,
    cache: dict[Any, Any],
    xp: Any,
) -> None:
    """Adapt grouped series spectra to the shared batched XMAC kernel.

    Spectral normalization and sample-exposure semantics stay with callers.
    Only the unnormalized cross-product sum is accumulated here.
    """
    if spectrum_a.ndim != 3 or spectrum_b.shape != spectrum_a.shape:
        raise ValueError("grouped spectra must have matching 3D shapes")
    groups, width, channels = map(int, spectrum_a.shape)
    if min(groups, width, channels) <= 0:
        raise ValueError("grouped spectra dimensions must be positive")
    if (
        output.ndim != 3
        or output.shape[0] != groups
        or output.shape[2] != channels
        or not 0 <= pair_index < output.shape[1]
        or not output.flags.c_contiguous
    ):
        raise ValueError("grouped output does not match spectra and pair")
    if groups > 65_535:
        for first in range(0, groups, 65_535):
            stop = first + 65_535
            accumulate_grouped_cross_power(
                spectrum_a[first:stop],
                spectrum_b[first:stop],
                output[first:stop],
                pair_index,
                cache=cache,
                xp=xp,
            )
        return
    pair_count = int(output.shape[1])
    key = (
        int(xp.cuda.Device().id),
        groups,
        width,
        channels,
        pair_count,
        pair_index,
    )
    if key not in cache:
        starts = np.arange(groups, dtype=np.int64) * width
        lengths = np.full(groups, width, dtype=np.int64)
        offsets = (
            np.arange(groups, dtype=np.int64) * pair_count + pair_index
        ) * channels
        cache[key] = (
            prepare_visibility_index(starts, device_dtype=xp.int32, xp=xp),
            prepare_visibility_index(lengths, device_dtype=xp.int32, xp=xp),
            prepare_visibility_index(offsets, device_dtype=xp.int64, xp=xp),
            prepare_visibility_index(
                np.arange(groups, dtype=np.int64),
                device_dtype=xp.int64,
                xp=xp,
            ),
            xp.ones(groups * width, dtype=xp.float64),
            xp.empty(groups, dtype=xp.float64),
        )
    starts, lengths, offsets, weight_offsets, weights, weight_sum = cache[key]
    batched_visibility_product(
        xp.ascontiguousarray(spectrum_a.reshape(groups * width, channels)),
        xp.ascontiguousarray(spectrum_b.reshape(groups * width, channels)),
        normalization=1.0,
        channels_to_average=1,
        weight_a=weights,
        group_starts=starts,
        group_lengths=lengths,
        output_offsets=offsets,
        weight_offsets=weight_offsets,
        output=output.reshape(-1),
        weight_sum=weight_sum,
        xp=xp,
        _validate_ranges=False,
        accumulate=True,
    )


def batched_visibility_products(
    spectra: list[Any],
    product_bands: list[tuple[int, int]],
    *,
    normalizations: list[float],
    channels_to_average: int,
    weight_pairs: list[tuple[Any, Any]],
    multiply_weights: list[bool],
    group_starts: Any,
    group_lengths: Any,
    output_offsets: Any,
    weight_offsets: Any,
    output: Any,
    weight_sum: Any,
    xp: Any,
    _validate_ranges: bool = True,
) -> None:
    """Accumulate up to eight compatible products while loading bands once."""

    product_count = len(product_bands)
    band_count = len(spectra)
    if not 2 <= product_count <= 8:
        raise ValueError("fused visibility requires two to eight products")
    if not 1 <= band_count <= 8:
        raise ValueError(
            "fused visibility requires one to eight unique bands"
        )
    if channels_to_average not in (1, 4, 32):
        raise ValueError(
            "fused visibility supports averaging factors 1, 4, and 32"
        )
    first_shape = spectra[0].shape
    if len(first_shape) != 2:
        raise ValueError("fused visibility spectra must be matrices")
    block_count, input_channels = map(int, first_shape)
    if not 0 < block_count <= np.iinfo(np.int32).max:
        raise ValueError("fused visibility block count is out of range")
    if not 0 < input_channels <= np.iinfo(np.int32).max:
        raise ValueError("fused visibility channel count is out of range")
    if input_channels % channels_to_average:
        raise ValueError("channel count is not divisible by averaging factor")
    output_channels = input_channels // channels_to_average
    spectrum_strides = []
    element_bytes = int(xp.dtype(xp.complex64).itemsize)
    for spectrum in spectra:
        if spectrum.dtype != xp.complex64 or spectrum.shape != first_shape:
            raise TypeError(
                "fused visibility spectra must be matching complex64 data"
            )
        if (
            int(spectrum.strides[1]) != element_bytes
            or int(spectrum.strides[0]) < input_channels * element_bytes
            or int(spectrum.strides[0]) % element_bytes
        ):
            raise TypeError(
                "fused visibility spectra must have contiguous channels "
                "and positive integral row strides"
            )
        spectrum_strides.append(int(spectrum.strides[0]) // element_bytes)
    for left, right in product_bands:
        if not 0 <= left < band_count or not 0 <= right < band_count:
            raise ValueError("fused visibility product band is out of range")
    if not (
        len(normalizations)
        == len(weight_pairs)
        == len(multiply_weights)
        == product_count
    ):
        raise ValueError(
            "fused visibility product metadata lengths do not match"
        )
    if any(not np.isfinite(value) or value <= 0 for value in normalizations):
        raise ValueError(
            "visibility normalizations must be finite and positive"
        )
    for left_weight, right_weight in weight_pairs:
        for weights in (left_weight, right_weight):
            if weights.dtype != xp.float64 or weights.shape != (block_count,):
                raise TypeError(
                    "fused visibility weights must match the block count"
                )
            if not weights.flags.c_contiguous:
                raise TypeError("fused visibility weights must be contiguous")
    group_count = int(
        group_starts.size
        if hasattr(group_starts, "size")
        else np.asarray(group_starts).size
    )
    if group_count <= 0 or group_count > 65_535:
        raise ValueError("fused visibility requires 1..65535 groups")
    group_starts_host, group_starts_device = _prepare_index_array(
        "group_starts", group_starts, (group_count,), xp.int32, xp
    )
    group_lengths_host, group_lengths_device = _prepare_index_array(
        "group_lengths", group_lengths, (group_count,), xp.int32, xp
    )
    expected_offsets_shape = (product_count, group_count)
    output_offsets_host, output_offsets_device = _prepare_index_array(
        "output_offsets",
        output_offsets,
        expected_offsets_shape,
        xp.int64,
        xp,
    )
    weight_offsets_host, weight_offsets_device = _prepare_index_array(
        "weight_offsets",
        weight_offsets,
        expected_offsets_shape,
        xp.int64,
        xp,
    )
    if output.dtype != xp.complex64 or output.ndim != 1:
        raise TypeError("fused visibility output must be a complex64 vector")
    if weight_sum.dtype != xp.float64 or weight_sum.ndim != 1:
        raise TypeError(
            "fused visibility weight output must be a float64 vector"
        )
    if not output.flags.c_contiguous or not weight_sum.flags.c_contiguous:
        raise TypeError("fused visibility outputs must be contiguous")
    if _validate_ranges:
        _validate_visibility_ranges(
            group_starts_host,
            group_lengths_host,
            output_offsets_host,
            weight_offsets_host,
            block_count=block_count,
            output_channels=output_channels,
            output_size=int(output.size),
            weight_size=int(weight_sum.size),
        )
    group_starts = _device_index_array(
        group_starts_host, group_starts_device, xp.int32, xp
    )
    group_lengths = _device_index_array(
        group_lengths_host, group_lengths_device, xp.int32, xp
    )
    output_offsets = _device_index_array(
        output_offsets_host, output_offsets_device, xp.int64, xp
    )
    weight_offsets = _device_index_array(
        weight_offsets_host, weight_offsets_device, xp.int64, xp
    )

    product_indices = np.asarray(product_bands, dtype=np.int32)
    metadata_dtype = np.dtype(
        [
            ("spectrum_pointers", np.uint64, (8,)),
            ("spectrum_strides", np.int64, (8,)),
            ("product_left", np.int32, (8,)),
            ("product_right", np.int32, (8,)),
            ("band_counts", np.int32, (1,)),
            ("product_counts", np.int32, (1,)),
            ("product_rows", np.int32, (8,)),
            ("weight_left", np.uint64, (8,)),
            ("weight_right", np.uint64, (8,)),
            ("multiply_weights", np.uint8, (8,)),
            ("inverse_normalizations", np.float64, (8,)),
        ],
        align=True,
    )
    host_metadata = np.zeros(1, dtype=metadata_dtype)
    host_metadata["spectrum_pointers"][0, :band_count] = [
        value.data.ptr for value in spectra
    ]
    host_metadata["spectrum_strides"][0, :band_count] = spectrum_strides
    host_metadata["product_left"][0, :product_count] = product_indices[:, 0]
    host_metadata["product_right"][0, :product_count] = product_indices[:, 1]
    host_metadata["band_counts"][0, 0] = band_count
    host_metadata["product_counts"][0, 0] = product_count
    host_metadata["product_rows"][0, :product_count] = np.arange(
        product_count, dtype=np.int32
    )
    host_metadata["weight_left"][0, :product_count] = [
        pair[0].data.ptr for pair in weight_pairs
    ]
    host_metadata["weight_right"][0, :product_count] = [
        pair[1].data.ptr for pair in weight_pairs
    ]
    host_metadata["multiply_weights"][0, :product_count] = multiply_weights
    host_metadata["inverse_normalizations"][0, :product_count] = (
        np.reciprocal(np.asarray(normalizations, dtype=np.float64))
    )
    device_metadata = xp.asarray(host_metadata.view(np.uint8).reshape(-1))

    def metadata_field(name: str, dtype: Any) -> Any:
        assert metadata_dtype.fields is not None
        field_dtype, offset = metadata_dtype.fields[name][:2]
        return device_metadata[offset : offset + field_dtype.itemsize].view(
            dtype
        )

    spectrum_pointers = metadata_field("spectrum_pointers", xp.uint64)
    spectrum_strides_device = metadata_field("spectrum_strides", xp.int64)
    product_left = metadata_field("product_left", xp.int32)
    product_right = metadata_field("product_right", xp.int32)
    band_counts = metadata_field("band_counts", xp.int32)
    product_counts = metadata_field("product_counts", xp.int32)
    product_rows = metadata_field("product_rows", xp.int32)
    weight_left_pointers = metadata_field("weight_left", xp.uint64)
    weight_right_pointers = metadata_field("weight_right", xp.uint64)
    multiply_weights_device = metadata_field("multiply_weights", xp.uint8)
    inverse_normalizations = metadata_field(
        "inverse_normalizations", xp.float64
    )
    channel_tiles = (input_channels + 31) // 32
    _multi_visibility_kernel(band_count, product_count)(
        (channel_tiles, group_count, 1),
        (256,),
        (
            spectrum_pointers,
            spectrum_strides_device,
            product_left,
            product_right,
            band_counts,
            product_counts,
            product_rows,
            weight_left_pointers,
            weight_right_pointers,
            multiply_weights_device,
            group_starts,
            group_lengths,
            output_offsets,
            weight_offsets,
            inverse_normalizations,
            np.int32(group_count),
            np.int32(input_channels),
            np.int32(channels_to_average),
            np.int32(0),
            output,
            weight_sum,
        ),
    )


def batched_visibility_product_packs(
    packs: list[dict[str, Any]],
    *,
    channels_to_average: int,
    group_starts: Any,
    group_lengths: Any,
    output_offsets: Any,
    weight_offsets: Any,
    output: Any,
    weight_sum: Any,
    xp: Any,
    _validate_ranges: bool = True,
) -> None:
    """Launch independent fused-product packs in one CUDA grid."""

    launch_limit = current_policy().xmac_packs_per_launch
    pack_count = len(packs)
    if not 1 <= pack_count <= 65_535:
        raise ValueError("fused visibility requires 1..65535 packs")
    if channels_to_average not in (1, 4, 32):
        raise ValueError(
            "fused visibility supports averaging factors 1, 4, and 32"
        )
    first_spectra = packs[0]["spectra"]
    if not first_spectra:
        raise ValueError("fused visibility pack has no spectra")
    first_shape = first_spectra[0].shape
    if len(first_shape) != 2:
        raise ValueError("fused visibility spectra must be matrices")
    block_count, input_channels = map(int, first_shape)
    if input_channels % channels_to_average:
        raise ValueError("channel count is not divisible by averaging factor")
    output_channels = input_channels // channels_to_average
    element_bytes = int(xp.dtype(xp.complex64).itemsize)
    metadata_dtype = np.dtype(
        [
            ("spectrum_pointers", np.uint64, (pack_count, 8)),
            ("spectrum_strides", np.int64, (pack_count, 8)),
            ("product_left", np.int32, (pack_count, 8)),
            ("product_right", np.int32, (pack_count, 8)),
            ("band_counts", np.int32, (pack_count,)),
            ("product_counts", np.int32, (pack_count,)),
            ("product_rows", np.int32, (pack_count, 8)),
            ("weight_left", np.uint64, (pack_count, 8)),
            ("weight_right", np.uint64, (pack_count, 8)),
            ("multiply_weights", np.uint8, (pack_count, 8)),
            ("inverse_normalizations", np.float64, (pack_count, 8)),
        ],
        align=True,
    )
    host_metadata = np.zeros(1, dtype=metadata_dtype)
    all_rows: list[int] = []
    for pack_index, pack in enumerate(packs):
        spectra = pack["spectra"]
        product_bands = pack["product_bands"]
        normalizations = pack["normalizations"]
        weight_pairs = pack["weight_pairs"]
        multiply_weights = pack["multiply_weights"]
        product_rows = pack["product_rows"]
        band_count = len(spectra)
        product_count = len(product_bands)
        if not 2 <= product_count <= 8:
            raise ValueError(
                "fused visibility pack requires two to eight products"
            )
        if not 1 <= band_count <= 8:
            raise ValueError(
                "fused visibility pack requires one to eight unique bands"
            )
        if not (
            len(normalizations)
            == len(weight_pairs)
            == len(multiply_weights)
            == len(product_rows)
            == product_count
        ):
            raise ValueError("fused visibility pack metadata lengths differ")
        strides: list[int] = []
        for spectrum in spectra:
            if (
                spectrum.dtype != xp.complex64
                or spectrum.shape != first_shape
            ):
                raise TypeError(
                    "fused visibility spectra must be matching complex64 data"
                )
            if (
                int(spectrum.strides[1]) != element_bytes
                or int(spectrum.strides[0]) < input_channels * element_bytes
                or int(spectrum.strides[0]) % element_bytes
            ):
                raise TypeError(
                    "fused visibility spectra must have contiguous channels"
                )
            strides.append(int(spectrum.strides[0]) // element_bytes)
        indices = np.asarray(product_bands, dtype=np.int32)
        if np.any(indices < 0) or np.any(indices >= band_count):
            raise ValueError("fused visibility product band is out of range")
        if any(
            not np.isfinite(value) or value <= 0 for value in normalizations
        ):
            raise ValueError(
                "visibility normalizations must be finite and positive"
            )
        for left_weight, right_weight in weight_pairs:
            for weights in (left_weight, right_weight):
                if (
                    weights.dtype != xp.float64
                    or weights.shape != (block_count,)
                    or not weights.flags.c_contiguous
                ):
                    raise TypeError(
                        "fused visibility weights must match the block count"
                    )
        rows = np.asarray(product_rows, dtype=np.int32)
        if np.any(rows < 0):
            raise ValueError("fused visibility product row is negative")
        all_rows.extend(int(value) for value in rows)
        host_metadata["spectrum_pointers"][0, pack_index, :band_count] = [
            value.data.ptr for value in spectra
        ]
        host_metadata["spectrum_strides"][0, pack_index, :band_count] = (
            strides
        )
        host_metadata["product_left"][0, pack_index, :product_count] = (
            indices[:, 0]
        )
        host_metadata["product_right"][0, pack_index, :product_count] = (
            indices[:, 1]
        )
        host_metadata["band_counts"][0, pack_index] = band_count
        host_metadata["product_counts"][0, pack_index] = product_count
        host_metadata["product_rows"][0, pack_index, :product_count] = rows
        host_metadata["weight_left"][0, pack_index, :product_count] = [
            pair[0].data.ptr for pair in weight_pairs
        ]
        host_metadata["weight_right"][0, pack_index, :product_count] = [
            pair[1].data.ptr for pair in weight_pairs
        ]
        host_metadata["multiply_weights"][0, pack_index, :product_count] = (
            multiply_weights
        )
        host_metadata["inverse_normalizations"][
            0, pack_index, :product_count
        ] = np.reciprocal(np.asarray(normalizations, dtype=np.float64))

    offset_shape = (
        output_offsets.host.shape
        if isinstance(output_offsets, _PreparedIndexArray)
        else (
            output_offsets.shape
            if hasattr(output_offsets, "shape")
            else np.asarray(output_offsets).shape
        )
    )
    if len(offset_shape) != 2:
        raise ValueError("fused visibility output offsets must be a matrix")
    row_count = int(offset_shape[0])
    if any(row >= row_count for row in all_rows):
        raise ValueError("fused visibility product row is out of range")
    group_count = int(
        group_starts.size
        if hasattr(group_starts, "size")
        else np.asarray(group_starts).size
    )
    if group_count <= 0 or group_count > 65_535:
        raise ValueError("fused visibility requires 1..65535 groups")
    group_starts_host, group_starts_device = _prepare_index_array(
        "group_starts", group_starts, (group_count,), xp.int32, xp
    )
    group_lengths_host, group_lengths_device = _prepare_index_array(
        "group_lengths", group_lengths, (group_count,), xp.int32, xp
    )
    output_offsets_host, output_offsets_device = _prepare_index_array(
        "output_offsets",
        output_offsets,
        (row_count, group_count),
        xp.int64,
        xp,
    )
    weight_offsets_host, weight_offsets_device = _prepare_index_array(
        "weight_offsets",
        weight_offsets,
        (row_count, group_count),
        xp.int64,
        xp,
    )
    if output.dtype != xp.complex64 or output.ndim != 1:
        raise TypeError("fused visibility output must be a complex64 vector")
    if weight_sum.dtype != xp.float64 or weight_sum.ndim != 1:
        raise TypeError("fused visibility weight output must be float64")
    if not output.flags.c_contiguous or not weight_sum.flags.c_contiguous:
        raise TypeError("fused visibility outputs must be contiguous")
    if _validate_ranges:
        _validate_visibility_ranges(
            group_starts_host,
            group_lengths_host,
            output_offsets_host,
            weight_offsets_host,
            block_count=block_count,
            output_channels=output_channels,
            output_size=int(output.size),
            weight_size=int(weight_sum.size),
        )
    device_metadata = xp.asarray(host_metadata.view(np.uint8).reshape(-1))

    def metadata_field(name: str, dtype: Any) -> Any:
        assert metadata_dtype.fields is not None
        field_dtype, offset = metadata_dtype.fields[name][:2]
        return device_metadata[offset : offset + field_dtype.itemsize].view(
            dtype
        )

    channel_tiles = (input_channels + 31) // 32
    arguments = (
        metadata_field("spectrum_pointers", xp.uint64),
        metadata_field("spectrum_strides", xp.int64),
        metadata_field("product_left", xp.int32),
        metadata_field("product_right", xp.int32),
        metadata_field("band_counts", xp.int32),
        metadata_field("product_counts", xp.int32),
        metadata_field("product_rows", xp.int32),
        metadata_field("weight_left", xp.uint64),
        metadata_field("weight_right", xp.uint64),
        metadata_field("multiply_weights", xp.uint8),
        _device_index_array(
            group_starts_host, group_starts_device, xp.int32, xp
        ),
        _device_index_array(
            group_lengths_host, group_lengths_device, xp.int32, xp
        ),
        _device_index_array(
            output_offsets_host, output_offsets_device, xp.int64, xp
        ),
        _device_index_array(
            weight_offsets_host, weight_offsets_device, xp.int64, xp
        ),
        metadata_field("inverse_normalizations", xp.float64),
        np.int32(group_count),
        np.int32(input_channels),
        np.int32(channels_to_average),
        output,
        weight_sum,
    )
    kernel = _multi_visibility_kernel(
        max(len(pack["spectra"]) for pack in packs),
        max(len(pack["product_bands"]) for pack in packs),
    )
    step = min(launch_limit or pack_count, pack_count)
    for pack_offset in range(0, pack_count, step):
        kernel(
            (channel_tiles, group_count, min(step, pack_count - pack_offset)),
            (256,),
            (*arguments[:-2], np.int32(pack_offset), *arguments[-2:]),
        )
