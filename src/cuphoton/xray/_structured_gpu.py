# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Single-owner CUDA structured roots; no import-time CUDA initialization."""

from __future__ import annotations

import operator
import threading
from importlib.resources import files
from typing import Any

import numpy as np
from numpy.typing import NDArray

from ._structured_native import _require_array


class StructuredRootsGPU:
    """One device batch at a time on the constructing thread and device."""

    def __init__(self, cp: Any) -> None:
        if int(cp.cuda.runtime.runtimeGetVersion()) // 1000 != 13:
            raise RuntimeError("The structured GPU backend requires CUDA 13")
        self.cp = cp
        self.owner = threading.get_ident()
        self.device = cp.cuda.runtime.getDevice()
        self.closed = False
        self.source = (
            files("cuphoton.xray")
            .joinpath("src/structured_roots.cu")
            .read_text(encoding="utf-8")
        )
        # CuPy14.1.1 appends -ftz=true. That option concerns FP32; this kernel
        # uses FP64 throughout. Do not claim an overridden -ftz=false policy.
        self.options = (
            "--std=c++17",
            "--fmad=false",
            "--prec-div=true",
            "--prec-sqrt=true",
        )
        self.kernel = cp.RawKernel(
            self.source, "rds_lane", options=self.options
        )
        self.kernel.compile()
        # Compile before acquiring an owned stream: failed initialization
        # leaves no stream or in-flight batch for the caller to drain.
        self.stream = cp.cuda.Stream(non_blocking=True)

    def allocate(self, degree: int, batch: int) -> dict[str, Any]:
        """Allocate scratch separately from optional kernel timing."""
        self._check_owner()
        degree, batch = operator.index(degree), operator.index(batch)
        if not 1 <= degree <= 4096 or not 1 <= batch <= 2147483647:
            raise ValueError(
                "Require degree1..4096 and a positive actual batch"
            )
        cp = self.cp
        return dict(
            state=cp.empty((6, degree + 1, batch), dtype=cp.float64),
            real=cp.empty((degree, batch), dtype=cp.float64),
            imag=cp.empty((degree, batch), dtype=cp.float64),
            stats=cp.empty((8, batch), dtype=cp.int32),
        )

    def __call__(
        self,
        descending_soa: Any,
        buffers: dict[str, Any],
        *,
        block: int = 32,
        max_sweeps: int = -1,
    ) -> tuple[Any, Any, Any]:
        """Launch on current stream, no host copy/fence or policy fallback.

        Input is contiguous FP64 (degree+1,batch), descending polynomial
        coefficients including the leading coefficient. LP inputs need
        [1,-prediction_coefficients]. Keep input/buffers alive through the
        caller's completion fence. Accept a lane only if stats[0]==0 and
        stats[1]==degree. Other lanes contain diagnostic partial roots.
        """
        self._check_owner()
        cp = self.cp
        if (
            descending_soa.ndim != 2
            or descending_soa.dtype != cp.float64
            or not descending_soa.flags.c_contiguous
        ):
            raise ValueError("Input must be contiguous FP64 (degree+1,batch)")
        degree, batch = descending_soa.shape[0] - 1, descending_soa.shape[1]
        block, max_sweeps = operator.index(block), operator.index(max_sweeps)
        if (
            not 1 <= degree <= 4096
            or not 1 <= batch <= 2147483647
            or block not in (32, 64, 128, 256)
            or not -1 <= max_sweeps <= 2147483647
        ):
            raise ValueError("Invalid degree, batch, block or sweep cap")
        expected = dict(
            state=((6, degree + 1, batch), cp.float64),
            real=((degree, batch), cp.float64),
            imag=((degree, batch), cp.float64),
            stats=((8, batch), cp.int32),
        )
        for name, (shape, dtype) in expected.items():
            value = buffers[name]
            if (
                value.shape != shape
                or value.dtype != dtype
                or not value.flags.c_contiguous
            ):
                raise ValueError("Unexpected buffer layout: " + name)
            if value.device.id != descending_soa.device.id:
                raise ValueError("All buffers must share the input device")
        if cp.cuda.runtime.getDevice() != descending_soa.device.id:
            raise ValueError("Current device must own the input and buffers")
        spans = [
            (int(value.data.ptr), int(value.data.ptr) + value.nbytes)
            for value in (
                descending_soa,
                *(buffers[name] for name in expected),
            )
        ]
        if any(
            max(a, c) < min(b, d)
            for i, (a, b) in enumerate(spans)
            for c, d in spans[i + 1 :]
        ):
            raise ValueError("Input/state/output buffers must not overlap")
        self.kernel(
            ((batch + block - 1) // block,),
            (block,),
            (
                descending_soa,
                buffers["state"],
                buffers["real"],
                buffers["imag"],
                buffers["stats"],
                cp.int32(degree),
                cp.int32(batch),
                cp.int32(max_sweeps),
            ),
        )
        return buffers["real"], buffers["imag"], buffers["stats"]

    def solve(
        self,
        descending: NDArray[np.float64],
        *,
        max_sweeps: int = -1,
    ) -> tuple[NDArray[np.complex128], NDArray[np.int32], NDArray[np.uint8]]:
        """Transfer, solve and drain one batch.

        Callers must check every row's status before using its roots.
        """
        self._check_owner()
        _require_array(descending, 2)
        batch, columns = descending.shape
        if not 1 <= batch <= 2147483647 or not 2 <= columns <= 4097:
            raise ValueError(
                "Require a positive batch and polynomial degree 1..4096"
            )
        cp = self.cp
        host_soa = np.ascontiguousarray(descending.T)
        soa = buffers = None
        try:
            with self.stream:
                soa = cp.asarray(host_soa)
                buffers = self.allocate(columns - 1, batch)
                self(soa, buffers, max_sweeps=max_sweeps)
                self.stream.synchronize()
                roots = np.empty((batch, columns - 1, 2), dtype=np.float64)
                roots[:, :, 0] = cp.asnumpy(buffers["real"].T)
                roots[:, :, 1] = cp.asnumpy(buffers["imag"].T)
                stats = cp.asnumpy(buffers["stats"].T)
        finally:
            # Input, scratch and destinations remain alive through this fence,
            # including launch, copy and cancellation failures in the owner.
            self.stream.synchronize()
        return (
            roots.view(np.complex128).reshape(batch, columns - 1),
            stats,
            np.zeros((batch, 256), dtype=np.uint8),
        )

    def close(self) -> None:
        """Drain the owned stream before the executor releases this solver."""
        self._check_owner(allow_closed=True)
        if not self.closed:
            self.stream.synchronize()
            self.closed = True

    def _check_owner(self, *, allow_closed: bool = False) -> None:
        if threading.get_ident() != self.owner:
            raise RuntimeError(
                "Structured GPU solver must remain on its owner thread"
            )
        if self.cp.cuda.runtime.getDevice() != self.device:
            raise RuntimeError(
                "Structured GPU solver must remain on its owner device"
            )
        if self.closed and not allow_closed:
            raise RuntimeError("Structured GPU solver is closed")
