# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Benchmark and verify the steady-state CuPy visibility reduction."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np

_DUAL_POL_PRODUCTS = ((0, 2), (0, 3), (1, 2), (1, 3))
_DISTINCT_PRODUCTS = ((0, 1), (2, 3), (4, 5), (6, 7))


def _product_bands(
    product_count: int, band_layout: str = "dual-pol"
) -> tuple[tuple[int, int], ...]:
    if band_layout not in {"dual-pol", "distinct"}:
        raise ValueError("band layout must be dual-pol or distinct")
    if product_count == 1:
        return ((0, 1),)
    if not 2 <= product_count <= len(_DUAL_POL_PRODUCTS):
        raise ValueError("product count must be between one and four")
    if band_layout == "dual-pol":
        if product_count == 2:
            return ((0, 2), (1, 3))
        return _DUAL_POL_PRODUCTS[:product_count]
    if band_layout == "distinct":
        return _DISTINCT_PRODUCTS[:product_count]
    raise AssertionError("unreachable band layout")


def _band_count(product_bands: tuple[tuple[int, int], ...]) -> int:
    return 1 + max(index for pair in product_bands for index in pair)


def _nonuniform_host_inputs(
    band_count: int, block_count: int, channel_count: int
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    """Build compact, deterministic data for the independent CPU oracle."""
    blocks = np.arange(block_count, dtype=np.float32)[:, np.newaxis]
    channels = np.arange(channel_count, dtype=np.float32)[np.newaxis, :]
    weight_blocks = np.arange(block_count, dtype=np.float64)
    spectra = []
    weights = []
    for band in range(band_count):
        real = (
            np.float32(0.375 * (band + 1))
            + np.float32(0.0625) * blocks
            + np.float32(0.00390625) * channels
            + np.float32(0.0009765625)
            * np.remainder(blocks + channels + band, 7)
        )
        imag = (
            np.float32(-0.21875 * (band + 1))
            + np.float32(0.03125) * blocks
            - np.float32(0.001953125) * channels
            + np.float32(0.00048828125)
            * np.remainder(3 * blocks + channels + band, 5)
        )
        spectra.append((real + np.complex64(1j) * imag).astype(np.complex64))
        weights.append(
            (
                np.float64(0.125 * (band + 1))
                + np.float64(0.015625) * weight_blocks
                + np.float64(0.001953125)
                * np.remainder(weight_blocks + band, 3)
            ).astype(np.float64)
        )
    return tuple(spectra), tuple(weights)


def _validate_geometry(
    group_count: int,
    group_length: int,
    channel_count: int,
    channels_to_average: int,
    iterations: int,
    warmups: int,
) -> None:
    if group_count <= 0 or group_length <= 0 or channel_count <= 0:
        raise ValueError("visibility geometry values must be positive")
    if group_count > 65_535:
        raise ValueError("visibility benchmark supports at most 65535 groups")
    if group_count * group_length > np.iinfo(np.int32).max:
        raise ValueError("visibility benchmark block count is out of range")
    if channels_to_average not in (1, 4):
        raise ValueError("visibility benchmark averaging must be one or four")
    if channel_count % channels_to_average:
        raise ValueError("channel count must be divisible by averaging")
    if iterations <= 0 or warmups < 0:
        raise ValueError(
            "iterations must be positive and warmups nonnegative"
        )


def _summary(values: list[float]) -> dict[str, float]:
    median = statistics.median(values)
    return {
        "median": median,
        "mad": statistics.median(abs(value - median) for value in values),
        "minimum": min(values),
        "maximum": max(values),
    }


def _timed_cuda(
    cupy: Any,
    operation: Any,
    *,
    iterations: int,
    warmups: int,
) -> list[float]:
    for _ in range(warmups):
        operation()
    cupy.cuda.get_current_stream().synchronize()
    values = []
    for _ in range(iterations):
        start = cupy.cuda.Event()
        stop = cupy.cuda.Event()
        start.record()
        operation()
        stop.record()
        stop.synchronize()
        values.append(float(cupy.cuda.get_elapsed_time(start, stop)))
    return values


def _production_timing_operation(operation: Any) -> Any:
    """Bind production's no-range-validation setting for timed launches."""

    def launch() -> None:
        operation(validate_ranges=False)

    return launch


def _expected_results(
    spectra: tuple[np.ndarray, ...],
    band_weights: tuple[np.ndarray, ...],
    product_bands: tuple[tuple[int, int], ...],
    normalizations: tuple[float, ...],
    *,
    group_starts: np.ndarray,
    group_lengths: np.ndarray,
    channels_to_average: int,
) -> tuple[np.ndarray, np.ndarray]:
    input_channels = spectra[0].shape[1]
    output_channels = input_channels // channels_to_average
    group_count = group_starts.size
    output = np.empty(
        (len(product_bands), group_count, output_channels),
        dtype=np.complex64,
    )
    weights = np.empty((len(product_bands), group_count), dtype=np.float64)
    for product, ((left, right), normalization) in enumerate(
        zip(product_bands, normalizations, strict=True)
    ):
        for group, (start, length) in enumerate(
            zip(group_starts, group_lengths, strict=True)
        ):
            stop = int(start + length)
            cross = spectra[left][int(start) : stop] * np.conj(
                spectra[right][int(start) : stop]
            )
            channel_sums = np.sum(cross, axis=0, dtype=np.complex64)
            averaged = np.sum(
                channel_sums.reshape(output_channels, channels_to_average),
                axis=1,
                dtype=np.complex64,
            )
            output[product, group] = np.asarray(
                averaged / np.float64(normalization * channels_to_average),
                dtype=np.complex64,
            )
            weights[product, group] = np.sum(
                band_weights[left][int(start) : stop]
                * band_weights[right][int(start) : stop],
                dtype=np.float64,
            )
    return output, weights


def _verify_nonuniform_correctness(
    cupy: Any,
    batched_visibility_product: Any,
    batched_visibility_products: Any,
    prepare_visibility_index: Any,
    product_bands: tuple[tuple[int, int], ...],
    normalizations: tuple[float, ...],
    channels_to_average: int,
) -> dict[str, Any]:
    """Compare a compact GPU launch with an independent NumPy oracle."""
    group_starts_host = np.asarray((1, 13, 29), dtype=np.int32)
    group_lengths_host = np.asarray((9, 11, 10), dtype=np.int32)
    block_count = 40
    input_channels = 64
    output_channels = input_channels // channels_to_average
    host_spectra, host_weights = _nonuniform_host_inputs(
        _band_count(product_bands), block_count, input_channels
    )
    spectra = [cupy.asarray(values) for values in host_spectra]
    weights = [cupy.asarray(values) for values in host_weights]
    group_starts = prepare_visibility_index(
        group_starts_host, device_dtype=cupy.int32, xp=cupy
    )
    group_lengths = prepare_visibility_index(
        group_lengths_host, device_dtype=cupy.int32, xp=cupy
    )
    group_count = group_starts_host.size
    output_offsets_host = np.arange(
        len(product_bands) * group_count, dtype=np.int64
    ).reshape(len(product_bands), group_count)
    output_offsets_host *= output_channels
    weight_offsets_host = np.arange(
        len(product_bands) * group_count, dtype=np.int64
    ).reshape(len(product_bands), group_count)
    prepared_output_offsets = output_offsets_host
    prepared_weight_offsets = weight_offsets_host
    if len(product_bands) == 1:
        prepared_output_offsets = output_offsets_host[0]
        prepared_weight_offsets = weight_offsets_host[0]
    output_offsets = prepare_visibility_index(
        prepared_output_offsets, device_dtype=cupy.int64, xp=cupy
    )
    weight_offsets = prepare_visibility_index(
        prepared_weight_offsets, device_dtype=cupy.int64, xp=cupy
    )
    output = cupy.full(
        len(product_bands) * group_count * output_channels,
        cupy.nan,
        dtype=cupy.complex64,
    )
    weight_output = cupy.full(
        len(product_bands) * group_count, cupy.nan, dtype=cupy.float64
    )

    if len(product_bands) == 1:
        left, right = product_bands[0]
        batched_visibility_product(
            spectra[left],
            spectra[right],
            normalization=normalizations[0],
            channels_to_average=channels_to_average,
            weight_a=weights[left],
            weight_b=weights[right],
            group_starts=group_starts,
            group_lengths=group_lengths,
            output_offsets=output_offsets,
            weight_offsets=weight_offsets,
            output=output,
            weight_sum=weight_output,
            xp=cupy,
            _validate_ranges=True,
        )
    else:
        batched_visibility_products(
            spectra,
            list(product_bands),
            normalizations=list(normalizations),
            channels_to_average=channels_to_average,
            weight_pairs=[
                (weights[left], weights[right])
                for left, right in product_bands
            ],
            multiply_weights=[True] * len(product_bands),
            group_starts=group_starts,
            group_lengths=group_lengths,
            output_offsets=output_offsets,
            weight_offsets=weight_offsets,
            output=output,
            weight_sum=weight_output,
            xp=cupy,
            _validate_ranges=True,
        )
    cupy.cuda.get_current_stream().synchronize()

    expected_output, expected_weights = _expected_results(
        host_spectra,
        host_weights,
        product_bands,
        normalizations,
        group_starts=group_starts_host,
        group_lengths=group_lengths_host,
        channels_to_average=channels_to_average,
    )
    actual_output = cupy.asnumpy(output).reshape(expected_output.shape)
    actual_weights = cupy.asnumpy(weight_output).reshape(
        expected_weights.shape
    )
    np.testing.assert_allclose(
        actual_output, expected_output, rtol=5e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        actual_weights, expected_weights, rtol=1e-13, atol=1e-13
    )
    return {
        "status": "PASS",
        "maximum_output_error": float(
            np.max(np.abs(actual_output - expected_output))
        ),
        "maximum_weight_error": float(
            np.max(np.abs(actual_weights - expected_weights))
        ),
        "weight_accumulation": "float64",
        "spectrum_accumulation": "complex64",
        "oracle": (
            "independent NumPy reduction over deterministic nonuniform data"
        ),
        "groups": int(group_count),
        "group_starts": group_starts_host.tolist(),
        "group_lengths": group_lengths_host.tolist(),
        "channels": input_channels,
    }


def _nonuniform_timing_inputs(
    cupy: Any, band_count: int, block_count: int, channel_count: int
) -> tuple[list[Any], list[Any]]:
    """Build deterministic, production-shaped nonuniform device inputs."""
    rng = cupy.random.default_rng(27)
    spectra = []
    weights = []
    shape = (block_count, channel_count)
    for _ in range(band_count):
        real = rng.standard_normal(shape, dtype=cupy.float32)
        imag = rng.standard_normal(shape, dtype=cupy.float32)
        spectra.append((real + 1j * imag).astype(cupy.complex64))
        weights.append(rng.random(block_count, dtype=cupy.float64))
    return spectra, weights


def _run(arguments: argparse.Namespace) -> dict[str, Any]:
    import cupy

    from cuphoton.xbeam.core.product_kernels import (
        _batched_visibility_kernel,
        _multi_visibility_kernel,
    )
    from cuphoton.xbeam.cupy_kernels import (
        batched_visibility_product,
        batched_visibility_products,
        prepare_visibility_index,
    )

    _validate_geometry(
        arguments.groups,
        arguments.group_length,
        arguments.channels,
        arguments.channels_to_average,
        arguments.iterations,
        arguments.warmups,
    )
    product_bands = _product_bands(arguments.products, arguments.band_layout)
    block_count = arguments.groups * arguments.group_length
    output_channels = arguments.channels // arguments.channels_to_average
    normalizations = tuple(
        float(1 << product) for product in range(arguments.products)
    )
    parity = _verify_nonuniform_correctness(
        cupy,
        batched_visibility_product,
        batched_visibility_products,
        prepare_visibility_index,
        product_bands,
        normalizations,
        arguments.channels_to_average,
    )
    spectra, weights = _nonuniform_timing_inputs(
        cupy,
        _band_count(product_bands),
        block_count,
        arguments.channels,
    )
    if len({int(spectrum.data.ptr) for spectrum in spectra}) != len(spectra):
        raise AssertionError(
            "timing spectra must use distinct device buffers"
        )
    group_starts_host = (
        np.arange(arguments.groups, dtype=np.int32) * arguments.group_length
    )
    group_lengths_host = np.full(
        arguments.groups, arguments.group_length, dtype=np.int32
    )
    group_starts = prepare_visibility_index(
        group_starts_host, device_dtype=cupy.int32, xp=cupy
    )
    group_lengths = prepare_visibility_index(
        group_lengths_host, device_dtype=cupy.int32, xp=cupy
    )
    if arguments.products == 1:
        output_offsets_host = np.arange(arguments.groups, dtype=np.int64)
        output_offsets_host *= output_channels
        weight_offsets_host = np.arange(arguments.groups, dtype=np.int64)
        output_offsets = prepare_visibility_index(
            output_offsets_host, device_dtype=cupy.int64, xp=cupy
        )
        weight_offsets = prepare_visibility_index(
            weight_offsets_host, device_dtype=cupy.int64, xp=cupy
        )
        output = cupy.empty(
            arguments.groups * output_channels, dtype=cupy.complex64
        )
        weight_output = cupy.empty(arguments.groups, dtype=cupy.float64)

        left, right = product_bands[0]

        def operation(*, validate_ranges: bool) -> None:
            batched_visibility_product(
                spectra[left],
                spectra[right],
                normalization=normalizations[0],
                channels_to_average=arguments.channels_to_average,
                weight_a=weights[left],
                weight_b=weights[right],
                group_starts=group_starts,
                group_lengths=group_lengths,
                output_offsets=output_offsets,
                weight_offsets=weight_offsets,
                output=output,
                weight_sum=weight_output,
                xp=cupy,
                _validate_ranges=validate_ranges,
            )

        averaging_mode = arguments.channels_to_average
        kernel = _batched_visibility_kernel(averaging_mode)
        operation_name = "batched_visibility_product"
    else:
        output_offsets_host = np.arange(
            arguments.products * arguments.groups, dtype=np.int64
        ).reshape(arguments.products, arguments.groups)
        output_offsets_host *= output_channels
        weight_offsets_host = np.arange(
            arguments.products * arguments.groups, dtype=np.int64
        ).reshape(arguments.products, arguments.groups)
        output_offsets = prepare_visibility_index(
            output_offsets_host, device_dtype=cupy.int64, xp=cupy
        )
        weight_offsets = prepare_visibility_index(
            weight_offsets_host, device_dtype=cupy.int64, xp=cupy
        )
        output = cupy.empty(
            arguments.products * arguments.groups * output_channels,
            dtype=cupy.complex64,
        )
        weight_output = cupy.empty(
            arguments.products * arguments.groups, dtype=cupy.float64
        )

        def operation(*, validate_ranges: bool) -> None:
            batched_visibility_products(
                spectra,
                list(product_bands),
                normalizations=list(normalizations),
                channels_to_average=arguments.channels_to_average,
                weight_pairs=[
                    (weights[left], weights[right])
                    for left, right in product_bands
                ],
                multiply_weights=[True] * arguments.products,
                group_starts=group_starts,
                group_lengths=group_lengths,
                output_offsets=output_offsets,
                weight_offsets=weight_offsets,
                output=output,
                weight_sum=weight_output,
                xp=cupy,
                _validate_ranges=validate_ranges,
            )

        kernel = _multi_visibility_kernel()
        operation_name = "batched_visibility_products"

    operation(validate_ranges=True)
    cupy.cuda.get_current_stream().synchronize()

    timings = _timed_cuda(
        cupy,
        _production_timing_operation(operation),
        iterations=arguments.iterations,
        warmups=arguments.warmups,
    )
    timing_summary = _summary(timings)
    device = cupy.cuda.Device()
    properties = cupy.cuda.runtime.getDeviceProperties(device.id)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode(errors="replace")
    memory_clock_khz = int(properties["memoryClockRate"])
    memory_bus_bits = int(properties["memoryBusWidth"])
    theoretical_gb_s = 2 * memory_clock_khz * 1000 * memory_bus_bits / 8 / 1e9
    spectrum_read_bytes = (
        len(spectra)
        * block_count
        * arguments.channels
        * np.dtype(np.complex64).itemsize
    )
    weight_read_bytes = (
        2 * arguments.products * block_count * np.dtype(np.float64).itemsize
    )
    output_write_bytes = (
        arguments.products
        * arguments.groups
        * output_channels
        * np.dtype(np.complex64).itemsize
    )
    minimum_device_bytes = (
        spectrum_read_bytes
        + weight_read_bytes
        + output_write_bytes
        + weight_output.nbytes
    )
    effective_gb_s = minimum_device_bytes / (timing_summary["median"] * 1e6)
    return {
        "metadata": {
            "gpu": device_name,
            "compute_capability": [
                int(properties["major"]),
                int(properties["minor"]),
            ],
            "cupy": cupy.__version__,
            "operation": operation_name,
            "groups": arguments.groups,
            "group_length": arguments.group_length,
            "channels": arguments.channels,
            "channels_to_average": arguments.channels_to_average,
            "products": arguments.products,
            "band_layout": arguments.band_layout,
            "unique_spectrum_buffers": len(spectra),
            "product_bands": [list(pair) for pair in product_bands],
            "iterations": arguments.iterations,
            "warmups": arguments.warmups,
            "timing_scope": (
                "steady-state public API GPU elapsed time with production "
                "range validation disabled"
            ),
            "timing_input": "deterministic nonuniform device data",
        },
        "parity": parity,
        "kernel": {
            "threads": 256,
            "registers_per_thread": int(kernel.num_regs),
            "static_shared_bytes": int(kernel.shared_size_bytes),
        },
        "traffic": {
            "minimum_device_bytes": minimum_device_bytes,
            "theoretical_memory_gb_s": theoretical_gb_s,
            "estimated_effective_gb_s": effective_gb_s,
        },
        "gpu_elapsed_milliseconds": timing_summary,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and benchmark the current CuPy visibility RawKernel. "
            "Defaults model 20 one-second integrations with 500 FFT "
            "blocks, 4096 channels, and four polarization products."
        )
    )
    parser.add_argument("--groups", type=int, default=20)
    parser.add_argument("--group-length", type=int, default=500)
    parser.add_argument("--channels", type=int, default=4096)
    parser.add_argument(
        "--channels-to-average", type=int, choices=(1, 4), default=1
    )
    parser.add_argument(
        "--products", type=int, choices=(1, 2, 3, 4), default=4
    )
    parser.add_argument(
        "--band-layout",
        choices=("dual-pol", "distinct"),
        default="dual-pol",
        help=(
            "dual-pol uses the representative four-buffer cross-baseline; "
            "distinct uses two buffers per product up to the eight-buffer "
            "fused ceiling"
        ),
    )
    parser.add_argument("--iterations", type=int, default=31)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    result = _run(arguments)
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if arguments.output is not None:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
