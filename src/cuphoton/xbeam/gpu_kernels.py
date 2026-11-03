# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Gpu kernels implementation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .core.gpu_products import (
    _device_index_array as _device_index_array,
)
from .core.gpu_products import (
    _prepare_index_array as _prepare_index_array,
)
from .core.gpu_products import (
    _PreparedIndexArray as _PreparedIndexArray,
)
from .core.gpu_products import (
    _validate_visibility_ranges as _validate_visibility_ranges,
)
from .core.gpu_products import (
    batched_visibility_product as batched_visibility_product,
)
from .core.gpu_products import (
    batched_visibility_product_packs as batched_visibility_product_packs,
)
from .core.gpu_products import (
    batched_visibility_products as batched_visibility_products,
)
from .core.gpu_products import (
    prepare_visibility_index as prepare_visibility_index,
)
from .core.gpu_products import (
    visibility_index_view as visibility_index_view,
)
from .core.gpu_products import (
    visibility_product as visibility_product,
)
from .kernel_compilation import (
    _channelize_residual_kernel,
    _clear_rejected_fft_kernel,
    _decode_kernel,
    _front_chunked_kernel,
    _front_chunked_real_bands_kernel,
    _front_kernel,
    _front_packed_real_bands_kernel,
    _front_weight_kernel,
    _residual_kernel,
)

BLOCK_PARAMETERS_DTYPE = np.dtype(
    [
        ("output_index", np.int64),
        ("coarse_sample", np.int64),
        ("block_start", np.float64),
        ("delay_zero", np.float64),
        ("delay_step", np.float64),
        ("delay_quadratic", np.float64),
        ("residual", np.float64),
        ("source_offset", np.int64),
        ("source_cutover", np.int64),
        ("valid_start", np.int64),
        ("valid_stop", np.int64),
    ],
    align=True,
)


FRONT_BAND_PARAMETERS_DTYPE = np.dtype(
    [
        ("source_start_sample", np.int64),
        ("channel_index", np.int32),
        ("_padding", np.int32),
        ("fringe_reference_hz", np.float64),
        ("lo_offset_hz", np.float64),
        ("wall_lo_cycles", np.float64),
    ],
    align=True,
)


def prepare_block_parameters(
    *,
    output_indices: Any,
    coarse_samples: Any,
    block_starts: Any,
    delay_zero: Any,
    delay_step: Any,
    delay_quadratic: Any,
    sample_rate_hz: float,
    centre_delay: Any | None = None,
    source_timing_offsets: Any | None = None,
    source_offsets: Any | None = None,
    source_cutovers: Any | None = None,
    valid_starts: Any | None = None,
    valid_stops: Any | None = None,
    xp: Any,
) -> Any:
    """Pack all per-block inputs and upload them in one transfer."""

    host_values = [
        np.asarray(output_indices, dtype=np.int64),
        np.asarray(coarse_samples, dtype=np.int64),
        np.asarray(block_starts, dtype=np.float64),
        np.asarray(delay_zero, dtype=np.float64),
        np.asarray(delay_step, dtype=np.float64),
        np.asarray(delay_quadratic, dtype=np.float64),
    ]
    if any(value.ndim != 1 for value in host_values):
        raise ValueError("block parameters must be one-dimensional")
    block_count = host_values[0].size
    if any(value.size != block_count for value in host_values[1:]):
        raise ValueError("block parameter arrays must have equal lengths")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be finite and positive")
    if centre_delay is None:
        centre_delay_host = np.zeros(block_count, dtype=np.float64)
    else:
        centre_delay_host = np.asarray(centre_delay, dtype=np.float64)
        if (
            centre_delay_host.ndim != 1
            or centre_delay_host.size != block_count
        ):
            raise ValueError("centre_delay must have one value per block")

    packed = np.empty(block_count, dtype=BLOCK_PARAMETERS_DTYPE)
    assert BLOCK_PARAMETERS_DTYPE.names is not None
    for name, value in zip(
        BLOCK_PARAMETERS_DTYPE.names[:6], host_values, strict=True
    ):
        packed[name] = value
    if source_timing_offsets is None:
        timing_offsets = np.zeros(block_count, dtype=np.int64)
    else:
        timing_offsets = np.asarray(source_timing_offsets)
        if (
            timing_offsets.shape != (block_count,)
            or timing_offsets.dtype.kind != "i"
        ):
            raise ValueError(
                "source timing offsets must be signed integers per block"
            )
    packed["residual"] = (
        host_values[1] - timing_offsets
    ) / sample_rate_hz - centre_delay_host
    if (source_offsets is None) != (source_cutovers is None):
        raise ValueError(
            "source offsets and cutovers must be provided together"
        )
    if source_offsets is None:
        packed["source_offset"] = 0
        packed["source_cutover"] = 0
    else:
        offsets = np.asarray(source_offsets)
        cutovers = np.asarray(source_cutovers)
        if (
            offsets.shape != (block_count,)
            or cutovers.shape != (block_count,)
            or offsets.dtype.kind != "i"
            or cutovers.dtype.kind != "i"
            or np.any(cutovers < 0)
        ):
            raise ValueError("invalid source offsets or cutovers")
        packed["source_offset"] = offsets
        packed["source_cutover"] = cutovers
    if (valid_starts is None) != (valid_stops is None):
        raise ValueError("valid starts and stops must be provided together")
    if valid_starts is None:
        packed["valid_start"] = 0
        packed["valid_stop"] = np.iinfo(np.int64).max
    else:
        starts = np.asarray(valid_starts)
        stops = np.asarray(valid_stops)
        if (
            starts.shape != (block_count,)
            or stops.shape != (block_count,)
            or starts.dtype.kind != "i"
            or stops.dtype.kind != "i"
            or np.any(starts < 0)
            or np.any(stops < starts)
        ):
            raise ValueError("invalid local sample-validity ranges")
        packed["valid_start"] = starts
        packed["valid_stop"] = stops
    return xp.asarray(packed.view(np.uint8))


def _block_count(block_parameters: Any) -> int:
    if block_parameters.dtype != np.dtype(np.uint8):
        raise TypeError("packed block parameters must be uint8 data")
    if not block_parameters.flags.c_contiguous:
        raise TypeError("packed block parameters must be contiguous")
    if block_parameters.size % BLOCK_PARAMETERS_DTYPE.itemsize:
        raise ValueError("packed block parameter buffer has an invalid size")
    return int(block_parameters.size // BLOCK_PARAMETERS_DTYPE.itemsize)


def _positive_kernel_integer(value: Any, name: str, maximum: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if not 0 < result <= maximum:
        raise ValueError(f"{name} is outside the supported kernel range")
    return result


def _validate_packed_geometry(
    *,
    samples_per_frame: Any,
    payload_bytes: Any,
    channel_count: Any,
    complex_input: bool,
) -> tuple[int, int, int]:
    maximum = int(np.iinfo(np.int32).max)
    samples_per_frame = _positive_kernel_integer(
        samples_per_frame, "samples_per_frame", maximum
    )
    payload_bytes = _positive_kernel_integer(
        payload_bytes, "payload_bytes", maximum
    )
    channel_count = _positive_kernel_integer(
        channel_count, "channel_count", maximum
    )
    components = 2 if complex_input else 1
    if payload_bytes * 4 != samples_per_frame * channel_count * components:
        raise ValueError(
            "packed VDIF geometry does not exactly fill each payload"
        )
    return samples_per_frame, payload_bytes, channel_count


def _validate_source_validity(
    source_validity: Any,
    validity_stride: int,
    source_size: int,
    packed_samples_per_frame: int = 0,
) -> int:
    validity_stride = _positive_kernel_integer(
        validity_stride, "validity_stride", int(np.iinfo(np.int32).max)
    )
    if (
        source_validity.dtype != np.dtype(np.uint8)
        or not source_validity.flags.c_contiguous
    ):
        raise TypeError("source validity must be contiguous uint8 data")
    stride = packed_samples_per_frame or validity_stride
    required = (source_size + stride - 1) // stride
    if source_validity.size < required:
        raise ValueError("source validity buffer is too short")
    return validity_stride


def _front_chunk_counts(
    *,
    fft_length: int,
    block_count: int,
    band_count: int,
    complex_input: bool,
    xp: Any,
) -> Any:
    """Allocate chunk counts only for qualified real-input batches."""

    if complex_input or fft_length != 262144 or not 8 <= block_count <= 1600:
        return None
    try:
        return xp.empty((band_count, block_count, 64), dtype=xp.uint32)
    except xp.cuda.memory.OutOfMemoryError:
        return None


def front_rotate(
    *,
    source: Any,
    source_validity: Any,
    validity_stride: int = 1,
    authoritative_validity: bool = False,
    source_start_sample: int,
    block_parameters: Any,
    fft_length: int,
    sample_rate_hz: float,
    fringe_reference_hz: float,
    lo_offset_hz: float,
    wall_lo_cycles: float,
    complex_input: bool,
    source_sample_count: int | None = None,
    packed_samples_per_frame: int = 0,
    packed_payload_bytes: int = 0,
    packed_channel_count: int = 0,
    packed_channel_index: int = 0,
    xp: Any,
    _validate_geometry: bool = True,
    _fft_batch_rows: int | None = None,
) -> tuple[Any, Any]:
    """Gather, mask, fringe-rotate, and reduce valid-sample weights."""

    block_count = _block_count(block_parameters)
    packed_input = source_sample_count is not None
    if packed_input:
        assert source_sample_count is not None
        if source.dtype != xp.uint8 or not source.flags.c_contiguous:
            raise TypeError(
                "packed VDIF payloads must be contiguous uint8 data"
            )
        if _validate_geometry:
            (
                packed_samples_per_frame,
                packed_payload_bytes,
                packed_channel_count,
            ) = _validate_packed_geometry(
                samples_per_frame=packed_samples_per_frame,
                payload_bytes=packed_payload_bytes,
                channel_count=packed_channel_count,
                complex_input=complex_input,
            )
            source_sample_count = _positive_kernel_integer(
                source_sample_count,
                "source_sample_count",
                int(np.iinfo(np.int64).max),
            )
            if isinstance(
                packed_channel_index, (bool, np.bool_)
            ) or not isinstance(packed_channel_index, (int, np.integer)):
                raise TypeError("packed_channel_index must be an integer")
            packed_channel_index = int(packed_channel_index)
            if not 0 <= packed_channel_index < packed_channel_count:
                raise ValueError("packed VDIF channel index is out of range")
            frame_count, partial_frame = divmod(
                source_sample_count, packed_samples_per_frame
            )
            if (
                partial_frame
                or source.size != frame_count * packed_payload_bytes
            ):
                raise ValueError(
                    "packed VDIF source size does not match "
                    "source_sample_count"
                )
        else:
            packed_samples_per_frame = int(packed_samples_per_frame)
            packed_payload_bytes = int(packed_payload_bytes)
            packed_channel_count = int(packed_channel_count)
            packed_channel_index = int(packed_channel_index)
            source_sample_count = int(source_sample_count)
        source_size = source_sample_count
    else:
        expected_dtype = xp.complex64 if complex_input else xp.float32
        if source.dtype != expected_dtype or not source.flags.c_contiguous:
            kind = "complex64" if complex_input else "float32"
            raise TypeError(
                f"decoded voltage samples must be contiguous {kind} data"
            )
        source_size = source.size
    if not authoritative_validity:
        validity_stride = _validate_source_validity(
            source_validity,
            validity_stride,
            source_size,
            packed_samples_per_frame if packed_input else 0,
        )
    batch_rows = block_count if _fft_batch_rows is None else _fft_batch_rows
    if batch_rows < block_count:
        raise ValueError("FFT batch capacity is smaller than the active rows")
    try:
        rotated = xp.empty((batch_rows, fft_length), dtype=xp.complex64)
    except xp.cuda.memory.OutOfMemoryError:
        if batch_rows == block_count:
            raise
        rotated = xp.empty((block_count, fft_length), dtype=xp.complex64)
    if rotated.shape[0] > block_count:
        rotated[block_count:].fill(0)
    weights = xp.empty(block_count, dtype=xp.float64)
    counts = _front_chunk_counts(
        fft_length=fft_length,
        block_count=block_count,
        band_count=1,
        complex_input=complex_input,
        xp=xp,
    )
    kernel = (
        _front_kernel(complex_input, packed_input)
        if counts is None
        else _front_chunked_kernel(packed_input)
    )
    kernel(
        (block_count,) if counts is None else (block_count, 1, 64),
        (256,),
        (
            source,
            source_validity,
            np.int32(validity_stride),
            np.int32(authoritative_validity),
            np.int64(source_size),
            np.int32(packed_samples_per_frame),
            np.int32(packed_payload_bytes),
            np.int32(packed_channel_count),
            np.int32(packed_channel_index),
            np.int64(source_start_sample),
            block_parameters,
            np.int32(fft_length),
            np.float64(1.0 / sample_rate_hz),
            np.float64(fringe_reference_hz),
            np.float64(lo_offset_hz),
            np.float64(wall_lo_cycles),
            rotated,
            weights if counts is None else counts,
        ),
    )
    if counts is not None:
        _front_weight_kernel()(
            (block_count,),
            (64,),
            (counts, np.int32(64), np.int32(fft_length), weights),
        )
    return rotated, weights


def front_rotate_packed_real_bands(
    *,
    source: Any,
    source_validity: Any,
    validity_stride: int,
    source_sample_count: int,
    block_parameters: Any,
    source_start_samples: Sequence[int],
    fringe_reference_hz: Sequence[float],
    lo_offset_hz: Sequence[float],
    wall_lo_cycles: Sequence[float],
    packed_channel_indices: Sequence[int],
    fft_length: int,
    sample_rate_hz: float,
    packed_samples_per_frame: int,
    packed_payload_bytes: int,
    packed_channel_count: int,
    xp: Any,
) -> tuple[Any, Any]:
    """Front-process several real channels sharing one packed VDIF payload."""

    if source.dtype != xp.uint8 or not source.flags.c_contiguous:
        raise TypeError("packed VDIF payloads must be contiguous uint8 data")
    band_count = len(packed_channel_indices)
    if not 1 < band_count <= 65_535:
        raise ValueError("packed band front-end requires 2..65535 bands")
    metadata_fields = (
        source_start_samples,
        fringe_reference_hz,
        lo_offset_hz,
        wall_lo_cycles,
    )
    if any(len(values) != band_count for values in metadata_fields):
        raise ValueError(
            "packed band front-end metadata lengths do not match"
        )
    total_blocks = _block_count(block_parameters)
    block_count, remainder = divmod(total_blocks, band_count)
    if remainder or block_count <= 0:
        raise ValueError(
            "packed band block parameters must contain equal nonempty batches"
        )
    fft_length = _positive_kernel_integer(
        fft_length, "fft_length", int(np.iinfo(np.int32).max)
    )
    source_sample_count = _positive_kernel_integer(
        source_sample_count,
        "source_sample_count",
        int(np.iinfo(np.int64).max),
    )
    (
        packed_samples_per_frame,
        packed_payload_bytes,
        packed_channel_count,
    ) = _validate_packed_geometry(
        samples_per_frame=packed_samples_per_frame,
        payload_bytes=packed_payload_bytes,
        channel_count=packed_channel_count,
        complex_input=False,
    )
    channel_indices = np.asarray(packed_channel_indices, dtype=np.int32)
    if np.any(channel_indices < 0) or np.any(
        channel_indices >= packed_channel_count
    ):
        raise ValueError("packed VDIF channel index is out of range")
    if not np.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be finite and positive")
    frame_count, partial_frame = divmod(
        source_sample_count, packed_samples_per_frame
    )
    if partial_frame or source.size != frame_count * packed_payload_bytes:
        raise ValueError(
            "packed VDIF source size does not match source_sample_count"
        )

    validity_stride = _validate_source_validity(
        source_validity,
        validity_stride,
        source_sample_count,
        packed_samples_per_frame,
    )

    metadata = np.zeros(band_count, dtype=FRONT_BAND_PARAMETERS_DTYPE)
    metadata["source_start_sample"] = np.asarray(
        source_start_samples, dtype=np.int64
    )
    metadata["channel_index"] = channel_indices
    metadata["fringe_reference_hz"] = np.asarray(
        fringe_reference_hz, dtype=np.float64
    )
    metadata["lo_offset_hz"] = np.asarray(lo_offset_hz, dtype=np.float64)
    metadata["wall_lo_cycles"] = np.asarray(wall_lo_cycles, dtype=np.float64)
    if not all(
        np.all(np.isfinite(metadata[name]))
        for name in (
            "fringe_reference_hz",
            "lo_offset_hz",
            "wall_lo_cycles",
        )
    ):
        raise ValueError("packed band phase metadata must be finite")
    device_metadata = xp.asarray(metadata.view(np.uint8).reshape(-1))
    rotated = xp.empty(
        (band_count, block_count, fft_length), dtype=xp.complex64
    )
    weights = xp.empty((band_count, block_count), dtype=xp.float64)
    counts = _front_chunk_counts(
        fft_length=fft_length,
        block_count=block_count,
        band_count=band_count,
        complex_input=False,
        xp=xp,
    )
    kernel = (
        _front_packed_real_bands_kernel()
        if counts is None
        else _front_chunked_real_bands_kernel()
    )
    kernel(
        (block_count, band_count)
        if counts is None
        else (block_count, band_count, 64),
        (256,),
        (
            source,
            source_validity,
            np.int32(validity_stride),
            np.int64(source_sample_count),
            np.int32(packed_samples_per_frame),
            np.int32(packed_payload_bytes),
            np.int32(packed_channel_count),
            device_metadata,
            block_parameters,
            np.int32(block_count),
            np.int32(fft_length),
            np.float64(1.0 / sample_rate_hz),
            rotated,
            weights if counts is None else counts,
        ),
    )
    if counts is not None:
        _front_weight_kernel()(
            (band_count * block_count,),
            (64,),
            (counts, np.int32(64), np.int32(fft_length), weights),
        )
    return rotated, weights


def decode_2bit_band(
    payloads: Any,
    *,
    frame_count: int,
    samples_per_frame: int,
    payload_bytes: int,
    channel_count: int,
    channel_index: int,
    complex_input: bool,
    xp: Any,
) -> Any:
    """Bulk-decode one logical channel from packed VDIF frame payloads."""

    if payloads.dtype != xp.uint8 or not payloads.flags.c_contiguous:
        raise TypeError("packed VDIF payloads must be contiguous uint8 data")
    frame_count = _positive_kernel_integer(
        frame_count, "frame_count", int(np.iinfo(np.int64).max)
    )
    samples_per_frame, payload_bytes, channel_count = (
        _validate_packed_geometry(
            samples_per_frame=samples_per_frame,
            payload_bytes=payload_bytes,
            channel_count=channel_count,
            complex_input=complex_input,
        )
    )
    if payloads.size != frame_count * payload_bytes:
        raise ValueError(
            "packed VDIF payload size does not match frame metadata"
        )
    if isinstance(channel_index, (bool, np.bool_)) or not isinstance(
        channel_index, (int, np.integer)
    ):
        raise TypeError("channel_index must be an integer")
    channel_index = int(channel_index)
    if not 0 <= channel_index < channel_count:
        raise ValueError("VDIF channel index is out of range")
    if frame_count > np.iinfo(np.int64).max // samples_per_frame:
        raise ValueError(
            "decoded sample count exceeds the kernel index range"
        )
    sample_count = frame_count * samples_per_frame
    dtype = xp.complex64 if complex_input else xp.float32
    output = xp.empty(sample_count, dtype=dtype)
    threads = 256
    blocks = (sample_count + threads - 1) // threads
    _decode_kernel(complex_input)(
        (blocks,),
        (threads,),
        (
            payloads,
            np.int64(sample_count),
            np.int32(samples_per_frame),
            np.int32(payload_bytes),
            np.int32(channel_count),
            np.int32(channel_index),
            output,
        ),
    )
    return output


def apply_residual_phase(
    spectrum: Any, block_parameters: Any, frequencies: Any, xp: Any
) -> Any:
    """Apply per-block residual delay phase to a spectrum in place."""

    if spectrum.dtype not in {
        xp.dtype(xp.complex64),
        xp.dtype(xp.complex128),
    }:
        raise TypeError(
            "RawKernel residual phase requires a complex spectrum"
        )
    if spectrum.ndim != 2 or spectrum.shape[0] != _block_count(
        block_parameters
    ):
        raise ValueError(
            "spectrum must have one row per packed parameter block"
        )
    if (
        frequencies.dtype != xp.float64
        or frequencies.shape != (spectrum.shape[1],)
        or not frequencies.flags.c_contiguous
    ):
        raise TypeError(
            "residual frequencies must be a contiguous float64 vector "
            "with one value per channel"
        )
    element_count = int(spectrum.size)
    threads = 256
    blocks = (element_count + threads - 1) // threads
    _residual_kernel(spectrum.dtype == xp.complex128)(
        (blocks,),
        (threads,),
        (
            spectrum,
            block_parameters,
            frequencies,
            np.int64(element_count),
            np.int32(spectrum.shape[-1]),
        ),
    )
    return spectrum


def channelize_and_apply_residual(
    transformed: Any,
    block_parameters: Any,
    *,
    mapping: str,
    channel_count: int,
    baseband_bin_offset: int,
    channel_spacing_hz: float,
    xp: Any,
) -> Any:
    """Map raw FFT bins and apply residual phase in one production kernel."""

    if transformed.dtype != xp.complex64 or transformed.ndim != 2:
        raise TypeError(
            "fused GPU channelization requires a complex64 matrix"
        )
    if not transformed.flags.c_contiguous:
        raise TypeError("raw FFT output must be contiguous")
    if transformed.shape[0] != _block_count(block_parameters):
        raise ValueError(
            "FFT output must have one row per packed parameter block"
        )
    fft_length = int(transformed.shape[1])
    if channel_count <= 0 or channel_count > fft_length:
        raise ValueError("channel_count must fit within the FFT output")
    if (
        mapping in {"complex_lsb", "complex_dsb"}
        and channel_count != fft_length
    ):
        raise ValueError(f"{mapping} requires one output channel per FFT bin")
    if not np.isfinite(channel_spacing_hz) or channel_spacing_hz <= 0:
        raise ValueError("channel_spacing_hz must be finite and positive")

    inplace = mapping == "identity" and channel_count == fft_length
    output = (
        transformed
        if inplace
        else xp.empty(
            (transformed.shape[0], channel_count), dtype=xp.complex64
        )
    )
    threads = 256
    _channelize_residual_kernel(mapping)(
        (int(transformed.shape[0]),),
        (threads,),
        (
            transformed,
            block_parameters,
            np.int32(fft_length),
            np.int32(channel_count),
            np.int32(baseband_bin_offset),
            np.float64(channel_spacing_hz),
            output,
        ),
    )
    return output


def clear_rejected_fft_rows(
    spectrum: Any,
    weight: Any,
    reported_weight: Any,
    rejected_rows: np.ndarray,
    *,
    xp: Any,
) -> bool:
    """Clear owned rows in one sparse launch; decline other layouts."""
    if (
        spectrum.dtype != xp.complex64
        or spectrum.ndim != 2
        or spectrum.strides[1] != spectrum.dtype.itemsize
        or weight.dtype != xp.float64
        or weight.shape != (spectrum.shape[0],)
        or not weight.flags.c_contiguous
        or (
            reported_weight is not None
            and (
                reported_weight.dtype != xp.float64
                or reported_weight.shape != weight.shape
                or not reported_weight.flags.c_contiguous
            )
        )
    ):
        return False
    rows = np.asarray(rejected_rows, dtype=np.int64)
    if rows.ndim != 1 or np.any(rows < 0) or np.any(rows >= weight.size):
        raise ValueError("rejected FFT rows are outside the station buffer")
    if rows.size:
        device_rows = xp.asarray(rows)
        _clear_rejected_fft_kernel()(
            (int(rows.size),),
            (256,),
            (
                spectrum,
                weight,
                weight if reported_weight is None else reported_weight,
                device_rows,
                np.int64(spectrum.strides[0] // spectrum.dtype.itemsize),
                np.int32(spectrum.shape[1]),
                np.bool_(reported_weight is not None),
            ),
        )
    return True
