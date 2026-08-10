# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from .linear_prediction import linear_prediction_numpy, synthetic_trace


@dataclass(frozen=True)
class SubspaceMethodBenchmark:
    method: str
    svd_backend: str
    samples: int
    model_order: int
    pencil_rows: int
    svd_rank: int
    singular_value_head: tuple[float, ...]
    rms_residual: float
    max_abs_reconstruction_diff: float
    elapsed_s: float
    roots_real: tuple[float, ...]
    roots_imag: tuple[float, ...]


@dataclass(frozen=True)
class SubspaceBenchmark:
    samples: int
    model_order: int
    components: int
    baseline_chi2: float
    baseline_rms_residual: float
    methods: tuple[SubspaceMethodBenchmark, ...]


@dataclass(frozen=True)
class _SubspaceSvd:
    u: np.ndarray
    singular_values: np.ndarray
    vh: np.ndarray
    backend: str
    rank: int


def compare_subspace_methods(
    time,
    trace,
    *,
    model_order: int,
    components: int,
    methods: tuple[str, ...] = ("matrix-pencil", "esprit"),
    svd_backends: tuple[str, ...] = ("full",),
    pencil_rows: int | None = None,
    randomized_oversamples: int = 8,
    power_iterations: int = 1,
    random_seed: int = 0,
) -> SubspaceBenchmark:
    """Compare NumPy subspace prototypes against current LPSVD output."""

    signal = np.asarray(trace, dtype=np.complex128)
    if signal.ndim != 1:
        raise ValueError("trace must be one-dimensional")
    if model_order <= 0:
        raise ValueError("model_order must be positive")

    baseline = linear_prediction_numpy(time, trace, components)
    baseline_residual = np.asarray(baseline.reconstruction) - np.asarray(
        trace
    )
    baseline_rms = float(np.sqrt(np.mean(np.abs(baseline_residual) ** 2)))

    results = []
    for svd_backend in svd_backends:
        for method in methods:
            start = perf_counter()
            roots, svd = _estimate_roots(
                signal,
                model_order=model_order,
                method=method,
                pencil_rows=pencil_rows,
                svd_backend=svd_backend,
                randomized_oversamples=randomized_oversamples,
                power_iterations=power_iterations,
                random_seed=random_seed,
            )
            amplitudes, reconstruction = _reconstruct_from_roots(
                signal,
                roots,
            )
            del amplitudes
            elapsed_s = perf_counter() - start
            residual = reconstruction - signal
            diff = reconstruction - np.asarray(baseline.reconstruction)
            rows = _resolved_pencil_rows(
                signal.shape[0],
                model_order=model_order,
                pencil_rows=pencil_rows,
            )
            sorted_roots = _sort_roots(roots)
            results.append(
                SubspaceMethodBenchmark(
                    method=method,
                    svd_backend=svd.backend,
                    samples=int(signal.shape[0]),
                    model_order=int(model_order),
                    pencil_rows=int(rows),
                    svd_rank=int(svd.rank),
                    singular_value_head=tuple(
                        float(value) for value in svd.singular_values[:5]
                    ),
                    rms_residual=float(
                        np.sqrt(np.mean(np.abs(residual) ** 2))
                    ),
                    max_abs_reconstruction_diff=float(np.max(np.abs(diff))),
                    elapsed_s=float(elapsed_s),
                    roots_real=tuple(
                        float(root.real) for root in sorted_roots
                    ),
                    roots_imag=tuple(
                        float(root.imag) for root in sorted_roots
                    ),
                )
            )

    return SubspaceBenchmark(
        samples=int(signal.shape[0]),
        model_order=int(model_order),
        components=int(components),
        baseline_chi2=float(baseline.chi2),
        baseline_rms_residual=baseline_rms,
        methods=tuple(results),
    )


def synthetic_subspace_benchmark(
    *,
    samples: int = 96,
    model_order: int = 5,
    components: int = 8,
    methods: tuple[str, ...] = ("matrix-pencil", "esprit"),
    svd_backends: tuple[str, ...] = ("full",),
) -> SubspaceBenchmark:
    time, trace = synthetic_trace(samples)
    return compare_subspace_methods(
        time,
        trace,
        model_order=model_order,
        components=components,
        methods=methods,
        svd_backends=svd_backends,
    )


def matrix_pencil_roots(
    trace,
    *,
    model_order: int,
    pencil_rows: int | None = None,
    svd_backend: str = "full",
):
    roots, _svd = _estimate_roots(
        np.asarray(trace, dtype=np.complex128),
        model_order=model_order,
        method="matrix-pencil",
        pencil_rows=pencil_rows,
        svd_backend=svd_backend,
        randomized_oversamples=8,
        power_iterations=1,
        random_seed=0,
    )
    return roots


def esprit_roots(
    trace,
    *,
    model_order: int,
    pencil_rows: int | None = None,
    svd_backend: str = "full",
):
    roots, _svd = _estimate_roots(
        np.asarray(trace, dtype=np.complex128),
        model_order=model_order,
        method="esprit",
        pencil_rows=pencil_rows,
        svd_backend=svd_backend,
        randomized_oversamples=8,
        power_iterations=1,
        random_seed=0,
    )
    return roots


def _estimate_roots(
    signal: np.ndarray,
    *,
    model_order: int,
    method: str,
    pencil_rows: int | None,
    svd_backend: str,
    randomized_oversamples: int,
    power_iterations: int,
    random_seed: int,
):
    if model_order <= 0:
        raise ValueError("model_order must be positive")
    h0, h1, rows = _hankel_pair(
        signal,
        model_order=model_order,
        pencil_rows=pencil_rows,
    )
    del rows
    svd = _subspace_svd(
        h0,
        requested=model_order,
        backend=svd_backend,
        randomized_oversamples=randomized_oversamples,
        power_iterations=power_iterations,
        random_seed=random_seed,
    )
    u = svd.u
    singular_values = svd.singular_values
    vh = svd.vh
    order = _effective_rank(
        singular_values,
        requested=model_order,
        rows=h0.shape[0],
        cols=h0.shape[1],
    )

    if method == "matrix-pencil":
        uk = u[:, :order]
        vk = vh.conj().T[:, :order]
        inv_s = np.diag(1.0 / singular_values[:order])
        pencil = uk.conj().T @ h1 @ vk @ inv_s
        return np.linalg.eigvals(pencil), _replace_svd_rank(svd, order)

    if method == "esprit":
        signal_space = u[:, :order]
        lower = signal_space[:-1, :]
        upper = signal_space[1:, :]
        phi = np.linalg.lstsq(lower, upper, rcond=None)[0]
        return np.linalg.eigvals(phi), _replace_svd_rank(svd, order)

    raise ValueError(f"unknown subspace method: {method}")


def _subspace_svd(
    matrix: np.ndarray,
    *,
    requested: int,
    backend: str,
    randomized_oversamples: int,
    power_iterations: int,
    random_seed: int,
) -> _SubspaceSvd:
    if backend == "full":
        u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
        return _SubspaceSvd(
            u=u,
            singular_values=singular_values,
            vh=vh,
            backend=backend,
            rank=int(min(matrix.shape)),
        )
    if backend == "randomized":
        return _randomized_svd(
            matrix,
            requested=requested,
            oversamples=randomized_oversamples,
            power_iterations=power_iterations,
            random_seed=random_seed,
        )
    if backend == "partial":
        return _partial_svd(
            matrix,
            requested=requested,
            random_seed=random_seed,
        )
    raise ValueError(f"unknown subspace SVD backend: {backend}")


def _partial_svd(
    matrix: np.ndarray,
    *,
    requested: int,
    random_seed: int,
) -> _SubspaceSvd:
    if requested <= 0:
        raise ValueError("requested SVD rank must be positive")

    min_dim = min(matrix.shape)
    if requested >= min_dim:
        u, singular_values, vh = np.linalg.svd(
            matrix,
            full_matrices=False,
        )
        return _SubspaceSvd(
            u=u,
            singular_values=singular_values,
            vh=vh,
            backend="partial",
            rank=int(min_dim),
        )

    try:
        from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, svds
    except ImportError as exc:
        raise ValueError("partial SVD requires scipy.sparse.linalg") from exc

    try:
        u, singular_values, vh = svds(
            matrix,
            k=requested,
            which="LM",
            return_singular_vectors=True,
            random_state=random_seed,
        )
    except (ArpackError, ArpackNoConvergence, RuntimeError) as exc:
        raise ValueError(f"partial SVD failed: {exc}") from exc
    descending = np.argsort(singular_values)[::-1]
    return _SubspaceSvd(
        u=u[:, descending],
        singular_values=singular_values[descending],
        vh=vh[descending, :],
        backend="partial",
        rank=int(requested),
    )


def _randomized_svd(
    matrix: np.ndarray,
    *,
    requested: int,
    oversamples: int,
    power_iterations: int,
    random_seed: int,
) -> _SubspaceSvd:
    if requested <= 0:
        raise ValueError("requested SVD rank must be positive")
    if oversamples < 0:
        raise ValueError("randomized_oversamples must be non-negative")
    if power_iterations < 0:
        raise ValueError("power_iterations must be non-negative")

    rows, cols = matrix.shape
    target_rank = min(max(requested + oversamples, requested), rows, cols)
    rng = np.random.default_rng(random_seed)
    omega = rng.standard_normal((cols, target_rank))
    if np.iscomplexobj(matrix):
        omega = omega + 1j * rng.standard_normal((cols, target_rank))

    sample = matrix @ omega
    for _iteration in range(power_iterations):
        sample, _unused = np.linalg.qr(sample, mode="reduced")
        sample = matrix @ (matrix.conj().T @ sample)
    q, _unused = np.linalg.qr(sample, mode="reduced")
    small = q.conj().T @ matrix
    u_small, singular_values, vh = np.linalg.svd(
        small,
        full_matrices=False,
    )
    u = q @ u_small
    return _SubspaceSvd(
        u=u[:, :target_rank],
        singular_values=singular_values[:target_rank],
        vh=vh[:target_rank, :],
        backend="randomized",
        rank=int(target_rank),
    )


def _replace_svd_rank(svd: _SubspaceSvd, rank: int) -> _SubspaceSvd:
    return _SubspaceSvd(
        u=svd.u,
        singular_values=svd.singular_values,
        vh=svd.vh,
        backend=svd.backend,
        rank=int(rank),
    )


def _hankel_pair(
    signal: np.ndarray,
    *,
    model_order: int,
    pencil_rows: int | None,
):
    if signal.ndim != 1:
        raise ValueError("trace must be one-dimensional")
    rows = _resolved_pencil_rows(
        signal.shape[0],
        model_order=model_order,
        pencil_rows=pencil_rows,
    )
    cols = signal.shape[0] - rows
    h0 = np.empty((rows, cols), dtype=np.complex128)
    h1 = np.empty((rows, cols), dtype=np.complex128)
    for row in range(rows):
        h0[row, :] = signal[row : row + cols]
        h1[row, :] = signal[row + 1 : row + cols + 1]
    return h0, h1, rows


def _resolved_pencil_rows(
    samples: int,
    *,
    model_order: int,
    pencil_rows: int | None,
):
    if samples < 4:
        raise ValueError("trace must contain at least 4 samples")
    rows = samples // 2 if pencil_rows is None else int(pencil_rows)
    cols = samples - rows
    if rows <= model_order or cols <= model_order:
        raise ValueError(
            "pencil shape must leave more rows and columns than model_order"
        )
    return rows


def _effective_rank(
    singular_values: np.ndarray,
    *,
    requested: int,
    rows: int,
    cols: int,
):
    finite = singular_values[np.isfinite(singular_values)]
    if finite.size == 0:
        raise ValueError("subspace SVD produced no finite singular values")
    max_s = float(finite[0])
    tolerance = np.finfo(np.float64).eps * max(rows, cols) * max_s
    rank = int(np.count_nonzero(finite > tolerance))
    if rank <= 0:
        raise ValueError("subspace SVD has no numerically supported rank")
    return min(int(requested), rank)


def _reconstruct_from_roots(signal: np.ndarray, roots: np.ndarray):
    sample_index = np.arange(signal.shape[0], dtype=np.float64)
    vandermonde = roots[None, :] ** sample_index[:, None]
    amplitudes = np.linalg.lstsq(vandermonde, signal, rcond=None)[0]
    reconstruction = vandermonde @ amplitudes
    return amplitudes, reconstruction


def _sort_roots(roots: np.ndarray):
    roots = np.asarray(roots, dtype=np.complex128)
    order = np.lexsort((roots.imag, roots.real, -np.abs(roots)))
    return roots[order]


__all__ = [
    "SubspaceBenchmark",
    "SubspaceMethodBenchmark",
    "compare_subspace_methods",
    "esprit_roots",
    "matrix_pencil_roots",
    "synthetic_subspace_benchmark",
]
