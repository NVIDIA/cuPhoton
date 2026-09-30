# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional Numba-CUDA-MLIR Gaussian kernels using CuPy-owned storage."""

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from typing import Any

import numpy as np

from ._types import FitMode

_THREADS = 128
_KERNEL_CACHE_SIZE = 16
_worker_dispatchers = threading.local()


def _kernels(
    cuda: Any, dtype: Any, height: int, width: int
) -> tuple[Any, Any]:
    """Construct independent dispatchers with their own launch state."""

    pixels = height * width
    libdevice = cuda.libdevice
    if dtype == np.float32:
        add = libdevice.fadd_rn
        multiply = libdevice.fmul_rn
        divide = libdevice.fdiv_rn
        power = libdevice.powf
    else:
        add = libdevice.dadd_rn
        multiply = libdevice.dmul_rn
        divide = libdevice.ddiv_rn
        power = libdevice.pow

    # These kernels need no experimental AST features. An explicit option
    # avoids optional-module auto-detection during concurrent startup.
    @cuda.jit(
        device=True,
        inline=True,
        fastmath=False,
        experimental_ast_transforms=False,
    )
    def components(parameters, row, observation):
        amplitude = parameters[row, 0]
        sigma_x = math.exp(parameters[row, 1])
        sigma_y = math.exp(parameters[row, 2])
        if not math.isfinite(sigma_x) or sigma_x <= 0:
            sigma_x = dtype(math.nan)
        if not math.isfinite(sigma_y) or sigma_y <= 0:
            sigma_y = dtype(math.nan)
        cosine = math.cos(parameters[row, 3])
        sine = math.sin(parameters[row, 3])
        inv_x2 = divide(dtype(1), multiply(sigma_x, sigma_x))
        inv_y2 = divide(dtype(1), multiply(sigma_y, sigma_y))
        pixel = observation % pixels
        x = add(dtype(pixel % width), -dtype((width - 1) / 2))
        y = add(dtype(pixel // width), -dtype((height - 1) / 2))
        xp = add(x, -parameters[row, 4])
        yp = add(y, -parameters[row, 5])
        xn = add(x, -parameters[row, 6])
        yn = add(y, -parameters[row, 7])
        px = add(multiply(cosine, xp), multiply(sine, yp))
        py = add(multiply(cosine, yp), -multiply(sine, xp))
        nx = add(multiply(cosine, xn), multiply(sine, yn))
        ny = add(multiply(cosine, yn), -multiply(sine, xn))
        pu = math.exp(
            multiply(
                dtype(-0.5),
                add(
                    multiply(multiply(px, px), inv_x2),
                    multiply(multiply(py, py), inv_y2),
                ),
            )
        )
        nu = math.exp(
            multiply(
                dtype(-0.5),
                add(
                    multiply(multiply(nx, nx), inv_x2),
                    multiply(multiply(ny, ny), inv_y2),
                ),
            )
        )
        positive = multiply(amplitude, pu)
        negative = multiply(amplitude, nu)
        sigma_x_cubed = dtype(power(sigma_x, dtype(3)))
        sigma_y_cubed = dtype(power(sigma_y, dtype(3)))
        p1 = divide(multiply(multiply(positive, px), px), sigma_x_cubed)
        p2 = divide(multiply(multiply(positive, py), py), sigma_y_cubed)
        p3 = multiply(
            multiply(multiply(-positive, px), py), add(inv_x2, -inv_y2)
        )
        p4 = multiply(
            positive,
            add(
                multiply(multiply(cosine, px), inv_x2),
                -multiply(multiply(sine, py), inv_y2),
            ),
        )
        p5 = multiply(
            positive,
            add(
                multiply(multiply(sine, px), inv_x2),
                multiply(multiply(cosine, py), inv_y2),
            ),
        )
        n1 = divide(multiply(multiply(negative, nx), nx), sigma_x_cubed)
        n2 = divide(multiply(multiply(negative, ny), ny), sigma_y_cubed)
        n3 = multiply(
            multiply(multiply(-negative, nx), ny), add(inv_x2, -inv_y2)
        )
        n4 = multiply(
            negative,
            add(
                multiply(multiply(cosine, nx), inv_x2),
                -multiply(multiply(sine, ny), inv_y2),
            ),
        )
        n5 = multiply(
            negative,
            add(
                multiply(multiply(sine, nx), inv_x2),
                multiply(multiply(cosine, ny), inv_y2),
            ),
        )
        plane = observation // pixels
        zero = dtype(0)
        if plane == 1:
            return (
                positive,
                pu,
                multiply(p1, sigma_x),
                multiply(p2, sigma_y),
                p3,
                p4,
                p5,
                zero,
                zero,
            )
        if plane == 2:
            return (
                negative,
                nu,
                multiply(n1, sigma_x),
                multiply(n2, sigma_y),
                n3,
                zero,
                zero,
                n4,
                n5,
            )
        return (
            add(positive, -negative),
            add(pu, -nu),
            multiply(add(p1, -n1), sigma_x),
            multiply(add(p2, -n2), sigma_y),
            add(p3, -n3),
            p4,
            p5,
            -n4,
            -n5,
        )

    @cuda.jit(fastmath=False, experimental_ast_transforms=False)
    def residual_kernel(parameters, indices, images, weights, residuals):
        row = cuda.blockIdx.x
        original = indices[row]
        for observation in range(cuda.threadIdx.x, images.shape[1], _THREADS):
            prediction = components(parameters, row, observation)[0]
            residuals[row, observation] = (
                prediction - images[original, observation]
            ) * weights[original, observation]

    @cuda.jit(fastmath=False, experimental_ast_transforms=False)
    def jacobian_kernel(parameters, indices, weights, jacobian):
        row = cuda.blockIdx.x
        original = indices[row]
        for observation in range(
            cuda.threadIdx.x, weights.shape[1], _THREADS
        ):
            sample = components(parameters, row, observation)
            weight = weights[original, observation]
            for parameter in range(8):
                jacobian[row, parameter, observation] = (
                    sample[parameter + 1] * weight
                )

    return residual_kernel, jacobian_kernel


def _cached_kernels(
    cuda: Any, dtype: Any, device: int, height: int, width: int
) -> tuple[Any, Any]:
    """Reuse a worker's dispatchers for stable dtype and image dimensions.

    Each thread retains at most sixteen device/shape/dtype specializations.
    Fit buffers and their stream lifetimes remain workspace-local.
    """

    cache = getattr(_worker_dispatchers, "cache", None)
    if cache is None:
        cache = OrderedDict()
        _worker_dispatchers.cache = cache
    key = (device, dtype.str, height, width)
    dispatchers = cache.get(key)
    if dispatchers is None:
        dispatchers = _kernels(cuda, dtype.type, height, width)
        cache[key] = dispatchers
        if len(cache) > _KERNEL_CACHE_SIZE:
            cache.popitem(last=False)
    cache.move_to_end(key)
    return dispatchers


class GaussianWorkspace:
    """Own one fit's buffers and owners, using worker-local dispatchers.

    Images and weights must already be sanitized and ready on CuPy's current
    stream. Every callback and its consumers must use that same stream. A
    workspace belongs to one fit and must not be called concurrently. Use it
    as a context manager: closing waits before releasing queued work's owners.
    The weighted Jacobian is scratch storage reused on that same stream.
    """

    def __init__(
        self,
        images: Any,
        weights: Any,
        *,
        image_shape: tuple[int, int],
        mode: FitMode,
    ) -> None:
        try:
            import cupy as cp
            from numba_cuda_mlir import cuda
        except ImportError as exc:
            raise ImportError(
                "Numba-CUDA-MLIR Gaussian kernels require CuPy and "
                "numba-cuda-mlir"
            ) from exc
        if mode not in {"difference", "split"}:
            raise ValueError("mode must be 'difference' or 'split'")
        height, width = image_shape
        if height < 1 or width < 1:
            raise ValueError("image dimensions must be positive")
        shape = (
            (height, width) if mode == "difference" else (3, height, width)
        )
        if images.ndim != len(shape) + 1 or images.shape[1:] != shape:
            raise ValueError("images do not match image_shape and mode")
        if weights.shape != images.shape or weights.dtype != images.dtype:
            raise ValueError("weights must have the images' shape and dtype")
        if images.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise TypeError("Numba-CUDA-MLIR supports float32 and float64")
        self._cp = cp
        self._cuda = cuda
        self._stream = cp.cuda.get_current_stream()
        self._device = cp.cuda.Device().id
        for value in (images, weights):
            if (
                not isinstance(value, cp.ndarray)
                or value.device.id != self._device
            ):
                raise ValueError(
                    "images and weights must be on the active GPU"
                )
            if not value.flags.c_contiguous:
                raise ValueError("images and weights must be contiguous")
        self._external_stream = cuda.external_stream(self._stream.ptr)
        self._images = images.reshape(images.shape[0], -1)
        self._weights = weights.reshape(weights.shape[0], -1)
        self._dtype = images.dtype
        self._jacobian = cp.empty(
            (images.shape[0], 8, self._images.shape[1]), dtype=self._dtype
        )
        self._stable_wrappers = {
            id(value): cuda.as_cuda_array(value, sync=False)
            for value in (self._images, self._weights, self._jacobian)
        }
        self._residual_kernel, self._jacobian_kernel = _cached_kernels(
            cuda, images.dtype, self._device, height, width
        )
        self._pending: list[tuple[Any, ...]] = []
        self._closed = False

    def _check(self, parameters: Any, indices: Any) -> None:
        if self._closed:
            raise RuntimeError("Gaussian workspace is closed")
        if (
            self._cp.cuda.Device().id != self._device
            or self._cp.cuda.get_current_stream().ptr != self._stream.ptr
        ):
            raise RuntimeError(
                "Gaussian workspace requires its original CUDA stream"
            )
        if (
            parameters.shape != (indices.size, 8)
            or parameters.dtype != self._dtype
            or indices.ndim != 1
            or indices.dtype != np.dtype("int64")
            or not 0 < indices.size <= self._images.shape[0]
        ):
            raise ValueError(
                "incompatible Gaussian parameters or row indices"
            )
        for value in (parameters, indices):
            if (
                value.device.id != self._device
                or not value.flags.c_contiguous
            ):
                raise ValueError(
                    "callback arrays must be contiguous on the active GPU"
                )
        if self._stream.done:
            self._pending.clear()

    def _launch(
        self, kernel: Any, rows: int, arrays: tuple[Any, ...]
    ) -> None:
        wrappers = tuple(
            self._stable_wrappers[id(value)]
            if id(value) in self._stable_wrappers
            else self._cuda.as_cuda_array(value, sync=False)
            for value in arrays
        )
        # Retain owners before enqueue, including when a launch raises.
        self._pending.append((arrays, wrappers))
        kernel[rows, _THREADS, self._external_stream](*wrappers)

    def residual(self, parameters: Any, *, indices: Any) -> Any:
        """Return fresh weighted residuals in compacted candidate order."""

        self._check(parameters, indices)
        result = self._cp.empty(
            (parameters.shape[0], self._images.shape[1]), dtype=self._dtype
        )
        self._launch(
            self._residual_kernel,
            parameters.shape[0],
            (parameters, indices, self._images, self._weights, result),
        )
        return result

    def normal_equations(
        self, parameters: Any, residuals: Any, *, indices: Any
    ) -> tuple[Any, Any]:
        """Return weighted gradient and Hessian in log-width coordinates."""

        self._check(parameters, indices)
        if (
            residuals.shape != (parameters.shape[0], self._images.shape[1])
            or residuals.dtype != self._dtype
            or residuals.device.id != self._device
            or not residuals.flags.c_contiguous
        ):
            raise ValueError("incompatible Gaussian residual array")
        self._launch(
            self._jacobian_kernel,
            parameters.shape[0],
            (parameters, indices, self._weights, self._jacobian),
        )
        jacobian = self._jacobian[: parameters.shape[0]]
        self._pending.append(((residuals, jacobian), ()))
        gradient = self._cp.einsum("knm,km->kn", jacobian, residuals)
        self._pending.append(((gradient,), ()))
        hessian = self._cp.einsum("knm,kpm->knp", jacobian, jacobian)
        self._pending.append(((hessian,), ()))
        return gradient, hessian

    def close(self) -> None:
        """Complete queued work before releasing any retained input owners."""

        if not self._closed:
            self._stream.synchronize()
            self._pending.clear()
            self._closed = True

    def __enter__(self) -> GaussianWorkspace:
        if self._closed:
            raise RuntimeError("Gaussian workspace is closed")
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
