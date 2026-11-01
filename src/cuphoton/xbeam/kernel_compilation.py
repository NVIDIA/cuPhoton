# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Kernel compilation implementation."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .core.product_kernels import (
    _batched_visibility_kernel as _batched_visibility_kernel,
)
from .core.product_kernels import (
    _multi_visibility_kernel as _multi_visibility_kernel,
)
from .core.product_kernels import (
    _visibility_kernel as _visibility_kernel,
)
from .cuda_channelize import (
    _CHANNELIZE_RESIDUAL_SOURCE,
    _CLEAR_REJECTED_FFT_SOURCE,
    _RESIDUAL_SOURCE,
)
from .cuda_front import _FRONT_SOURCE
from .cuda_ingest import _DECODE_SOURCE, _VDIF_INGEST_SOURCE
from .profiling import nvtx_range
from .runtime.cuda import _raw_function
from .runtime.cuda import _raw_module as _raw_module


@dataclass(frozen=True, slots=True)
class _CudaLaunchPolicy:
    front_threads: int = 256
    channelize_threads: int = 256
    ingest_threads: int = 256
    visibility_threads: int = 256


def _launch_policy_for_compute_major(major: int) -> _CudaLaunchPolicy:
    """Return conservative launch sizes selected by compute major version."""

    if major >= 9:
        # Hopper's larger SM and scheduler sustain the simple bandwidth-bound
        # header strip at 512 threads. Reduction kernels retain eight warps
        # because their shared-memory layout is specialized for that shape.
        return _CudaLaunchPolicy(ingest_threads=512)
    return _CudaLaunchPolicy()


@lru_cache(maxsize=16)
def _cuda_launch_policy(device_identifier: int, xp: Any) -> _CudaLaunchPolicy:
    properties = xp.cuda.runtime.getDeviceProperties(device_identifier)
    major = int(properties.get("major", properties.get(b"major", 0)))
    return _launch_policy_for_compute_major(major)


@lru_cache(maxsize=4)
def _front_kernel(complex_input: bool, packed_input: bool) -> Any:
    name = "front_"
    if packed_input:
        name += "packed_"
    name += "complex" if complex_input else "real"
    return _raw_function(_FRONT_SOURCE, name)


@lru_cache(maxsize=1)
def _front_packed_real_bands_kernel() -> Any:
    return _raw_function(_FRONT_SOURCE, "front_packed_real_bands")


@lru_cache(maxsize=2)
def _decode_kernel(complex_input: bool) -> Any:
    name = "decode_complex_band" if complex_input else "decode_real_band"
    return _raw_function(_DECODE_SOURCE, name)


@lru_cache(maxsize=2)
def _vdif_ingest_kernel(streaming: bool = False) -> Any:
    return _raw_function(
        _VDIF_INGEST_SOURCE,
        (
            "strip_scatter_vdif_payload_validity"
            if streaming
            else "strip_scatter_vdif_payload"
        ),
    )


@lru_cache(maxsize=1)
def _sparse_vdif_ingest_kernel() -> Any:
    return _raw_function(
        _VDIF_INGEST_SOURCE, "strip_scatter_sparse_vdif_payload"
    )


@lru_cache(maxsize=2)
def _residual_kernel(double_precision: bool) -> Any:
    return _raw_function(
        _RESIDUAL_SOURCE,
        "apply_residual_double"
        if double_precision
        else "apply_residual_float",
    )


@lru_cache(maxsize=4)
def _channelize_residual_kernel(mapping: str) -> Any:
    if mapping not in {"identity", "real_lsb", "complex_lsb", "complex_dsb"}:
        raise ValueError(f"unsupported GPU channel mapping {mapping!r}")
    return _raw_function(
        _CHANNELIZE_RESIDUAL_SOURCE, "channelize_residual_" + mapping
    )


@lru_cache(maxsize=32)
def warm_gpu_pipeline(
    device_identifier: int,
    complex_inputs: tuple[bool, ...],
    packed_input: bool,
    channel_mappings: tuple[str, ...],
    fft_lengths: tuple[int, ...],
    include_vdif_ingest: bool,
) -> None:
    """Initialize the job-specific CUDA modules while the CPU plans ingest.

    CuPy's on-disk cache avoids repeated NVRTC compilation, but every new
    process must still initialize CUDA and load the cached modules.  Calling
    this from a background planning task hides that latency.  The tiny FFTs
    initialize cuFFT's lazy-loaded code; production-shape plans remain owned
    by the correlation thread's normal plan cache.
    """

    with nvtx_range("xbeam:gpu-warmup"):
        import cupy
        from cupyx.scipy.fft import fft as cupy_fft

        with cupy.cuda.Device(device_identifier):
            with nvtx_range("xbeam:cuda-context-init"):
                cupy.cuda.runtime.free(0)
            with nvtx_range("xbeam:kernel-module-load"):
                if include_vdif_ingest:
                    _vdif_ingest_kernel()
                for complex_input in complex_inputs:
                    _front_kernel(complex_input, packed_input)
                for mapping in channel_mappings:
                    _channelize_residual_kernel(mapping)
                _multi_visibility_kernel()
            with nvtx_range("xbeam:cufft-initialize"):
                stream = cupy.cuda.Stream(non_blocking=True)
                with stream:
                    for fft_length in fft_lengths:
                        values = cupy.zeros(
                            (1, fft_length), dtype=cupy.complex64
                        )
                        cupy_fft(values, axis=-1, overwrite_x=True)
                stream.synchronize()


@lru_cache(maxsize=1)
def _clear_rejected_fft_kernel() -> Any:
    return _raw_function(
        _CLEAR_REJECTED_FFT_SOURCE, "clear_rejected_fft_rows"
    )
