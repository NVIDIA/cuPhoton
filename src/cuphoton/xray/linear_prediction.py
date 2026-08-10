# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from time import perf_counter
from typing import Any

import numpy as np

_CUPY_INSTALL_HINT = (
    " Run 'uv sync --extra gpu' for development or install 'cuphoton[gpu]'."
)


@dataclass(frozen=True)
class LinearPredictionResult:
    """One fitted linear-prediction trace and its modal decomposition.

    Attributes
    ----------
    backend
        ``"cpu"`` or ``"gpu"`` implementation used for the fit.
    time
        One-dimensional fitted sample coordinates in the input time units.
    time_components
        Array shaped ``(samples, modes)`` in the input trace units.
    reconstruction
        One-dimensional modal sum plus fitted constant, in trace units.
    frequency
        One-dimensional spectrum grid in cycles per input time unit.
    spectrum_components
        Modal spectra shaped ``(frequencies, modes)`` in trace-units times
        input-time-units.
    spectrum_total
        Modal sum on ``frequency`` in the same spectrum units.
    angular_frequency
        Selected modal angular frequencies in radians per input time unit.
    decay
        Selected exponential decay rates in inverse input time units.
    amplitude
        Selected modal amplitudes in input trace units.
    phase
        Selected modal phases in radians.
    chi2
        Mean squared reconstruction residual in squared trace units. This is
        retained under its historical name and is not variance-normalized.
    selected_model_order, decaying_root_count
        Selected SVD order and number of decaying prediction roots.
    singular_values
        Singular values of the prediction Hankel matrix.
    roots_stats
        Optional root-solver execution metadata.
    elapsed_s
        End-to-end fit time in seconds.
    """

    backend: str
    time: np.ndarray
    time_components: np.ndarray
    reconstruction: np.ndarray
    frequency: np.ndarray
    spectrum_components: np.ndarray
    spectrum_total: np.ndarray
    angular_frequency: np.ndarray
    decay: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    chi2: float
    selected_model_order: int
    decaying_root_count: int
    singular_values: np.ndarray
    roots_stats: PredictionRootsStats | None
    elapsed_s: float


@dataclass(frozen=True)
class LinearPredictionModes:
    """Selected modal rates on a NumPy- or CuPy-compatible array module.

    ``decay`` is in inverse input time units, ``angular_frequency`` is in
    radians per input time unit, and both arrays have shape ``(modes,)``.
    """

    decay: Any
    angular_frequency: Any
    decaying_root_count: int


@dataclass(frozen=True)
class LinearPredictionModeBatch:
    decay: Any
    angular_frequency: Any
    mode_counts: Any
    decaying_root_counts: Any


@dataclass(frozen=True)
class LinearPredictionComparison:
    """CPU/GPU fit pair and reconstruction differences in trace units."""

    cpu: LinearPredictionResult
    gpu: LinearPredictionResult | None
    max_abs_reconstruction_diff: float | None
    rms_reconstruction_diff: float | None


@dataclass(frozen=True)
class LinearPredictionP1BatchBenchmark:
    samples: int
    traces: int
    components: int
    repeat: int
    fit: LinearPredictionFitStats
    cpu_serial_best_s: float
    gpu_serial_best_s: float | None
    gpu_batched_best_s: float | None
    gpu_batch_speedup: float | None
    max_abs_coefficient_diff: float | None
    max_abs_eigenvalue_diff: float | None
    gpu_error: str | None


@dataclass(frozen=True)
class LinearPredictionP2Benchmark:
    samples: int
    traces: int
    components: int
    modes: int
    design_columns: int
    repeat: int
    fit: LinearPredictionFitStats
    cpu_serial_best_s: float
    gpu_serial_best_s: float | None
    gpu_batched_best_s: float | None
    gpu_batch_speedup: float | None
    max_abs_reconstruction_diff: float | None
    rms_reconstruction_diff: float | None
    gpu_error: str | None


@dataclass(frozen=True)
class LinearPredictionSavgolBenchmark:
    samples: int
    traces: int
    window_length: int
    polyorder: int
    repeat: int
    cpu_serial_best_s: float
    gpu_serial_best_s: float | None
    gpu_batched_best_s: float | None
    gpu_batch_speedup: float | None
    max_abs_filter_diff: float | None
    rms_filter_diff: float | None
    gpu_error: str | None


@dataclass(frozen=True)
class LinearPredictionFixedStagesBenchmark:
    samples: int
    traces: int
    components: int
    modes: int
    design_columns: int
    window_length: int
    polyorder: int
    repeat: int
    fit: LinearPredictionFitStats
    cpu_serial_best_s: float
    gpu_serial_best_s: float | None
    gpu_batched_best_s: float | None
    gpu_batch_speedup: float | None
    max_abs_filter_diff: float | None
    rms_filter_diff: float | None
    max_abs_reconstruction_diff: float | None
    rms_reconstruction_diff: float | None
    gpu_error: str | None


@dataclass(frozen=True)
class LinearPredictionModeGroup:
    mode_count: int
    trace_count: int
    design_columns: int


@dataclass(frozen=True)
class LinearPredictionVariableP2Benchmark:
    samples: int
    traces: int
    components: int
    batched_solver: str
    window_length: int
    polyorder: int
    repeat: int
    fits: tuple[LinearPredictionFitStats, ...]
    mode_count_min: int
    mode_count_max: int
    mode_count_unique: tuple[int, ...]
    mode_groups: tuple[LinearPredictionModeGroup, ...]
    max_design_columns: int
    padded_design_entries: int
    grouped_design_entries: int
    padding_overhead_ratio: float
    mode_reference_elapsed_s: float
    cpu_serial_best_s: float
    gpu_serial_best_s: float | None
    gpu_batched_best_s: float | None
    gpu_batch_speedup: float | None
    max_abs_reconstruction_diff: float | None
    rms_reconstruction_diff: float | None
    gpu_error: str | None


@dataclass(frozen=True)
class LinearPredictionVariableStagesBenchmark:
    samples: int
    traces: int
    components: int
    batched_solver: str
    window_length: int
    polyorder: int
    repeat: int
    fits: tuple[LinearPredictionFitStats, ...]
    mode_count_min: int
    mode_count_max: int
    mode_count_unique: tuple[int, ...]
    mode_groups: tuple[LinearPredictionModeGroup, ...]
    max_design_columns: int
    padded_design_entries: int
    grouped_design_entries: int
    padding_overhead_ratio: float
    mode_reference_elapsed_s: float
    cpu_serial_best_s: float
    gpu_serial_best_s: float | None
    gpu_batched_best_s: float | None
    gpu_batch_speedup: float | None
    max_abs_filter_diff: float | None
    rms_filter_diff: float | None
    max_abs_reconstruction_diff: float | None
    rms_reconstruction_diff: float | None
    gpu_error: str | None


@dataclass(frozen=True)
class LinearPredictionVariableArtifactsBenchmark:
    samples: int
    traces: int
    components: int
    batched_solver: str
    window_length: int
    polyorder: int
    repeat: int
    fits: tuple[LinearPredictionFitStats, ...]
    mode_count_min: int
    mode_count_max: int
    mode_count_unique: tuple[int, ...]
    mode_groups: tuple[LinearPredictionModeGroup, ...]
    max_design_columns: int
    mode_reference_elapsed_s: float
    cpu_serial_best_s: float
    gpu_serial_best_s: float | None
    gpu_batched_best_s: float | None
    gpu_batch_speedup: float | None
    max_abs_filter_diff: float | None
    rms_filter_diff: float | None
    max_abs_coefficient_diff: float | None
    max_abs_amplitude_diff: float | None
    max_abs_phase_diff: float | None
    max_abs_frequency_center_diff: float | None
    max_abs_time_component_diff: float | None
    max_abs_reconstruction_diff: float | None
    rms_reconstruction_diff: float | None
    max_abs_spectrum_component_diff: float | None
    max_abs_spectrum_total_diff: float | None
    max_abs_chi2_diff: float | None
    gpu_error: str | None


@dataclass(frozen=True)
class LinearPredictionRuntimeBridgeBenchmark:
    samples: int
    tiles: int
    rows_per_tile: int
    traces: int
    components: int
    batched_solver: str
    window_length: int
    polyorder: int
    repeat: int
    fits: tuple[LinearPredictionFitStats, ...]
    mode_count_min: int
    mode_count_max: int
    mode_count_unique: tuple[int, ...]
    mode_groups: tuple[LinearPredictionModeGroup, ...]
    max_design_columns: int
    p1_reference_elapsed_s: float
    gpu_serial_per_tile_best_s: float | None
    gpu_row_batched_per_tile_best_s: float | None
    gpu_multi_tile_grouped_best_s: float | None
    row_batched_speedup: float | None
    multi_tile_speedup: float | None
    multi_vs_row_batched_speedup: float | None
    max_abs_frequency_center_diff: float | None
    max_abs_time_component_diff: float | None
    max_abs_reconstruction_diff: float | None
    max_abs_amplitude_diff: float | None
    max_abs_phase_diff: float | None
    max_abs_chi2_diff: float | None
    gpu_error: str | None


@dataclass(frozen=True)
class PredictionRootsStats:
    """Execution metadata for one prediction-polynomial root solve.

    Timing is seconds; ``matrix_size`` is the companion-matrix order and
    ``row_count`` is the number of independently solved coefficient rows.
    """

    backend: str
    array_module: str
    matrix_size: int
    row_count: int
    failures: int
    elapsed_s: float


@dataclass(frozen=True)
class PredictionRootsBackendBenchmark:
    backend: str
    array_module: str
    batched: bool
    matrix_size: int
    row_count: int
    failures: int
    best_s: float | None
    max_abs_root_diff: float | None
    error: str | None


@dataclass(frozen=True)
class PredictionRootsBenchmark:
    samples: int
    traces: int
    components: int
    repeat: int
    fit: LinearPredictionFitStats
    backends: tuple[PredictionRootsBackendBenchmark, ...]


@dataclass(frozen=True)
class LinearPredictionFitStats:
    """Compact fit-quality and model-order summary.

    ``chi2`` is the historical mean squared residual in squared trace units;
    ``rms_residual`` is in input trace units.
    """

    requested_components: int
    selected_model_order: int
    matrix_size: int
    root_count: int
    selected_root_count: int
    decaying_root_count: int
    filtered_root_count: int
    chi2: float
    rms_residual: float


@dataclass(frozen=True)
class ModelOrderSweepEntry:
    components: int
    selected_model_order: int
    matrix_size: int
    root_count: int
    selected_root_count: int
    decaying_root_count: int
    filtered_root_count: int
    chi2: float
    rms_residual: float
    reconstruction_rms_error: float
    singular_value_head: tuple[float, ...]
    singular_value_tail: tuple[float, ...]
    singular_value_ratio: float | None
    elapsed_s: float


@dataclass(frozen=True)
class ModelOrderSweep:
    samples: int
    roots_backend: str
    relative_tolerance: float
    best_components: int
    best_selected_model_order: int
    best_rms_residual: float
    best_reconstruction_rms_error: float
    entries: tuple[ModelOrderSweepEntry, ...]


@dataclass(frozen=True)
class _PredictionRootsResult:
    roots: Any
    stats: PredictionRootsStats


@dataclass(frozen=True)
class _P1CoefficientsResult:
    singular_values: Any
    coefficients: Any
    selected_model_order: int


@dataclass(frozen=True)
class _P1Result:
    singular_values: Any
    coefficients: Any
    eigenvalues: Any
    selected_model_order: Any
    roots_stats: PredictionRootsStats | None = None


@dataclass(frozen=True)
class _P2Result:
    coefficients: Any
    reconstruction: Any


@dataclass(frozen=True)
class LinearPredictionP2Artifacts:
    coefficients: Any
    amplitude: Any
    phase: Any
    time_components: Any
    reconstruction: Any
    frequency: Any
    spectrum_components: Any
    spectrum_total: Any
    frequency_centers: Any
    chi2: Any


_P2Artifacts = LinearPredictionP2Artifacts


def amplitudes_and_phases_numpy(a0, a1):
    """Return amplitudes and phases from fitted coefficients."""

    a0_array = np.asarray(a0)
    a1_array = np.asarray(a1)
    dtype = np.result_type(a0_array, a1_array, float)
    amplitudes = np.zeros(a0_array.shape, dtype=dtype)
    phases = np.zeros(a0_array.shape, dtype=dtype)

    mask0 = a0_array == 0
    mask1 = a1_array == 0
    both_zero = mask0 & mask1
    a0_zero = mask0 & ~mask1
    a1_zero = ~mask0 & mask1
    both_nonzero = ~mask0 & ~mask1

    np.putmask(
        amplitudes,
        both_nonzero,
        np.sqrt(a0_array**2 + a1_array**2),
    )
    np.putmask(phases, both_nonzero, np.arctan2(a1_array, a0_array))

    np.putmask(amplitudes, a1_zero, np.abs(a0_array))
    np.putmask(phases, a1_zero, (1 - np.sign(a0_array)) * pi / 2)

    np.putmask(amplitudes, a0_zero, np.abs(a1_array))
    np.putmask(phases, a0_zero, np.sign(a1_array) * pi / 2)

    np.putmask(amplitudes, both_zero, 0)
    np.putmask(phases, both_zero, 0)

    return amplitudes, phases


def amplitudes_and_phases_cupy(a0: Any, a1: Any):
    """CuPy equivalent of :func:`amplitudes_and_phases_numpy`."""

    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "CuPy is required for GPU amplitude/phase calculation."
            + _CUPY_INSTALL_HINT
        ) from exc

    a0_array = cp.asarray(a0)
    a1_array = cp.asarray(a1)
    dtype = cp.result_type(a0_array, a1_array, float)
    amplitudes = cp.zeros(a0_array.shape, dtype=dtype)
    phases = cp.zeros(a0_array.shape, dtype=dtype)

    mask0 = a0_array == 0
    mask1 = a1_array == 0
    both_zero = mask0 & mask1
    a0_zero = mask0 & ~mask1
    a1_zero = ~mask0 & mask1
    both_nonzero = ~mask0 & ~mask1

    cp.putmask(
        amplitudes,
        both_nonzero,
        cp.sqrt(a0_array**2 + a1_array**2),
    )
    cp.putmask(phases, both_nonzero, cp.arctan2(a1_array, a0_array))

    cp.putmask(amplitudes, a1_zero, cp.abs(a0_array))
    cp.putmask(phases, a1_zero, (1 - cp.sign(a0_array)) * pi / 2)

    cp.putmask(amplitudes, a0_zero, cp.abs(a1_array))
    cp.putmask(phases, a0_zero, cp.sign(a1_array) * pi / 2)

    cp.putmask(amplitudes, both_zero, 0)
    cp.putmask(phases, both_zero, 0)

    return amplitudes, phases


__all__ = [
    "LinearPredictionComparison",
    "LinearPredictionFixedStagesBenchmark",
    "LinearPredictionFitStats",
    "LinearPredictionModeGroup",
    "LinearPredictionModeBatch",
    "LinearPredictionModes",
    "LinearPredictionP1BatchBenchmark",
    "LinearPredictionP2Benchmark",
    "LinearPredictionP2Artifacts",
    "LinearPredictionSavgolBenchmark",
    "LinearPredictionResult",
    "LinearPredictionVariableP2Benchmark",
    "LinearPredictionVariableArtifactsBenchmark",
    "LinearPredictionRuntimeBridgeBenchmark",
    "ModelOrderSweep",
    "ModelOrderSweepEntry",
    "PredictionRootsBackendBenchmark",
    "PredictionRootsBenchmark",
    "PredictionRootsStats",
    "amplitudes_and_phases_cupy",
    "amplitudes_and_phases_numpy",
    "benchmark_linear_prediction_fixed_stages",
    "benchmark_linear_prediction_p1_batch",
    "benchmark_linear_prediction_p2",
    "benchmark_linear_prediction_savgol",
    "benchmark_linear_prediction_variable_p2",
    "benchmark_linear_prediction_variable_artifacts",
    "benchmark_linear_prediction_runtime_bridge",
    "benchmark_prediction_roots",
    "compare_cpu_gpu",
    "linear_prediction_p2_artifacts_numpy",
    "linear_prediction_batched_legacy_rows_cupy",
    "linear_prediction_batched_legacy_tiles_cupy",
    "linear_prediction_legacy_rows_from_artifacts_cupy",
    "linear_prediction_legacy_rows_from_artifacts_numpy",
    "linear_prediction_mode_batch_from_roots_cupy",
    "linear_prediction_mode_batch_from_roots_numpy",
    "linear_prediction_modes_from_roots_cupy",
    "linear_prediction_modes_from_roots_numpy",
    "linear_prediction_variable_artifacts_cupy_batched",
    "linear_prediction_cupy",
    "linear_prediction_numpy",
    "model_order_sweep",
    "synthetic_prediction_coefficients",
    "synthetic_trace",
    "synthetic_trace_batch",
]


def synthetic_trace(samples: int = 96):
    """Build a deterministic decaying-sinusoid trace for smoke tests."""

    if samples < 16:
        raise ValueError("samples must be at least 16")
    time = np.linspace(0.0, 9.5, samples, dtype=np.float64)
    trace = (
        1.25 * np.exp(-0.09 * time) * np.cos(2.4 * time + 0.30)
        + 0.45 * np.exp(-0.03 * time) * np.cos(0.9 * time - 0.75)
        + 0.15
    )
    return time, trace


def synthetic_trace_batch(samples: int = 96, traces: int = 16):
    """Build deterministic trace rows shaped like one tiled ROI."""

    if traces < 1:
        raise ValueError("traces must be positive")
    time, _trace = synthetic_trace(samples)
    rows = []
    for idx in range(traces):
        phase = 0.30 + 0.025 * idx
        slow_phase = -0.75 + 0.013 * idx
        fast_frequency = 2.4 + 0.009 * idx
        slow_frequency = 0.9 + 0.006 * idx
        offset = 0.15 + 0.002 * idx
        row = (
            1.25
            * np.exp(-0.09 * time)
            * np.cos(fast_frequency * time + phase)
            + 0.45
            * np.exp(-0.03 * time)
            * np.cos(slow_frequency * time + slow_phase)
            + offset
        )
        rows.append(row)
    return time, np.asarray(rows, dtype=np.float64)


def synthetic_prediction_coefficients(
    *,
    samples: int = 800,
    traces: int = 16,
    n_components: int = 30,
) -> np.ndarray:
    """Build production-shaped prediction coefficient vectors.

    The coefficients come from the existing synthetic row traces after the P1
    SVD projection. This keeps the roots benchmark focused on root solving
    while still feeding it vectors shaped like the production fit.
    """

    time, trace_rows = synthetic_trace_batch(samples=samples, traces=traces)
    results = [
        _linear_prediction_p1_coefficients_impl(
            xp=np,
            time=time,
            trace=trace,
            n_components=n_components,
        )
        for trace in trace_rows
    ]
    return np.stack([item.coefficients for item in results])


def linear_prediction_numpy(
    time,
    trace,
    n_components: int,
    *,
    roots_backend: str = "eigvals",
):
    """Run the linear-prediction fit on NumPy arrays."""

    start = perf_counter()
    result = _linear_prediction_impl(
        xp=np,
        backend="cpu",
        time=time,
        trace=trace,
        n_components=n_components,
        roots_backend=roots_backend,
    )
    return _with_elapsed(result, perf_counter() - start)


def linear_prediction_cupy(
    time,
    trace,
    n_components: int,
    *,
    roots_backend: str = "eigvals",
):
    """Run the CuPy linear-prediction fit and return NumPy output."""

    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "CuPy is required for GPU linear prediction." + _CUPY_INSTALL_HINT
        ) from exc

    start = perf_counter()
    result = _linear_prediction_impl(
        xp=cp,
        backend="gpu",
        time=time,
        trace=trace,
        n_components=n_components,
        roots_backend=roots_backend,
    )
    cp.cuda.Stream.null.synchronize()
    return _with_elapsed(result, perf_counter() - start)


def linear_prediction_modes_from_roots_numpy(
    time,
    roots,
    singular_value_count: int,
) -> LinearPredictionModes:
    """Select current XRay P2 modes from P1 roots on NumPy arrays."""

    return _linear_prediction_modes_from_roots(
        np,
        time=time,
        roots=roots,
        singular_value_count=singular_value_count,
    )


def linear_prediction_modes_from_roots_cupy(
    time,
    roots,
    singular_value_count: int,
) -> LinearPredictionModes:
    """Select current XRay P2 modes from P1 roots on CuPy arrays."""

    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "CuPy is required for GPU mode selection from roots."
            + _CUPY_INSTALL_HINT
        ) from exc

    return _linear_prediction_modes_from_roots(
        cp,
        time=time,
        roots=roots,
        singular_value_count=singular_value_count,
    )


def linear_prediction_mode_batch_from_roots_numpy(
    time,
    root_rows,
    singular_value_counts,
) -> LinearPredictionModeBatch:
    """Select and pad current XRay P2 modes from NumPy P1 root rows."""

    return _linear_prediction_mode_batch_from_roots(
        np,
        time=time,
        root_rows=root_rows,
        singular_value_counts=singular_value_counts,
    )


def linear_prediction_mode_batch_from_roots_cupy(
    time,
    root_rows,
    singular_value_counts,
) -> LinearPredictionModeBatch:
    """Select and pad current XRay P2 modes from CuPy P1 root rows."""

    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "CuPy is required for GPU batch mode selection from roots."
            + _CUPY_INSTALL_HINT
        ) from exc

    return _linear_prediction_mode_batch_from_roots(
        cp,
        time=time,
        root_rows=root_rows,
        singular_value_counts=singular_value_counts,
    )


def linear_prediction_p2_artifacts_numpy(
    time,
    trace,
    decay,
    angular_frequency,
) -> LinearPredictionP2Artifacts:
    """Run production P2 artifact construction on NumPy arrays.

    The caller supplies the already-selected decay and angular-frequency
    modes. This keeps root/model-order behavior identical to the caller's P1
    path while exposing the P2 artifact surface used for production parity.
    """

    return _linear_prediction_p2_artifacts_impl(
        xp=np,
        time=time,
        trace=trace,
        decay=decay,
        angular_frequency=angular_frequency,
    )


def linear_prediction_variable_artifacts_cupy_batched(
    time,
    traces,
    decay,
    angular_frequency,
    mode_counts,
    *,
    solver: str = "grouped-pinv",
    window_length: int | None = None,
    polyorder: int | None = None,
) -> tuple[Any, LinearPredictionP2Artifacts]:
    """Batch production P2 artifact construction on CuPy arrays.

    ``decay`` and ``angular_frequency`` are padded arrays containing the modes
    selected by the existing P1/root path, with ``mode_counts`` recording how
    many entries in each row are active. Passing ``window_length`` and
    ``polyorder`` applies the same axis-wise Savitzky-Golay filtering used by
    the accepted variable-artifact workbench before the batched P2 solve.
    """

    if (window_length is None) != (polyorder is None):
        raise ValueError(
            "window_length and polyorder must be provided together"
        )
    if solver not in ("pinv", "grouped-pinv"):
        raise ValueError("solver must be 'pinv' or 'grouped-pinv'")

    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "CuPy is required for batched P2 artifact construction."
            + _CUPY_INSTALL_HINT
        ) from exc

    gpu_traces = cp.asarray(traces, dtype=cp.float64)
    if window_length is None:
        filtered = gpu_traces
    else:
        from cupyx.scipy.signal import savgol_filter as cupy_savgol_filter

        filtered = _savgol_cupy_batched(
            gpu_traces,
            window_length=window_length,
            polyorder=polyorder,
            savgol_filter=cupy_savgol_filter,
        )

    artifacts = _variable_p2_artifacts_cupy_batched_solver(
        cp.asarray(time, dtype=cp.float64),
        filtered,
        cp.asarray(decay, dtype=cp.float64),
        cp.asarray(angular_frequency, dtype=cp.float64),
        cp.asarray(mode_counts, dtype=cp.int64),
        solver=solver,
    )
    return filtered, artifacts


def linear_prediction_batched_legacy_rows_cupy(
    time,
    traces,
    root_rows,
    singular_value_rows,
    *,
    solver: str = "grouped-pinv",
    window_length: int | None = None,
    polyorder: int | None = None,
) -> tuple[Any, tuple[tuple[Any, ...], ...]]:
    """Batch post-P1 GPU work and return legacy ``lpfi`` row tuples.

    ``root_rows`` and ``singular_value_rows`` are the existing per-row P1
    outputs. This helper preserves that root/model-order behavior, batches
    only the optional Savitzky-Golay filtering plus P2 artifact construction,
    and converts the result back to the reference wrapper's 13-slot ``lpfi``
    rows.
    """

    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "CuPy is required for GPU legacy row batching."
            + _CUPY_INSTALL_HINT
        ) from exc

    gpu_time = cp.asarray(time, dtype=cp.float64)
    gpu_traces = cp.asarray(traces, dtype=cp.float64)
    root_rows = tuple(cp.asarray(row) for row in root_rows)
    singular_value_rows = tuple(
        cp.asarray(row, dtype=cp.float64) for row in singular_value_rows
    )
    if len(root_rows) != len(singular_value_rows):
        raise ValueError(
            "root_rows and singular_value_rows must have same length"
        )
    if gpu_time.ndim != 1:
        raise ValueError("time must be one-dimensional")
    if gpu_traces.ndim != 2:
        raise ValueError("traces must be two-dimensional")
    if gpu_traces.shape[0] != len(root_rows):
        raise ValueError("traces must have one row per P1 row")
    if gpu_traces.shape[1] != gpu_time.shape[0]:
        raise ValueError("traces must have one column per time sample")

    modes = linear_prediction_mode_batch_from_roots_cupy(
        gpu_time,
        root_rows,
        singular_value_rows,
    )
    filtered, artifacts = linear_prediction_variable_artifacts_cupy_batched(
        gpu_time,
        gpu_traces,
        modes.decay,
        modes.angular_frequency,
        modes.mode_counts,
        solver=solver,
        window_length=window_length,
        polyorder=polyorder,
    )
    legacy_rows = linear_prediction_legacy_rows_from_artifacts_cupy(
        gpu_time,
        artifacts,
        modes.decay,
        modes.angular_frequency,
        modes.mode_counts,
        singular_value_rows,
    )
    return filtered, legacy_rows


def linear_prediction_batched_legacy_tiles_cupy(
    time,
    tile_traces,
    tile_root_rows,
    tile_singular_value_rows,
    *,
    solver: str = "grouped-pinv",
    window_length: int | None = None,
    polyorder: int | None = None,
):
    """Batch post-P1 legacy rows across multiple runtime tiles.

    This keeps each row's existing P1 roots and singular values, flattens rows
    from several tiles into one grouped P2/artifact solve, then splits the
    legacy ``lpfi`` rows back into the original tile boundaries.
    """

    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "CuPy is required for GPU legacy tile batching."
            + _CUPY_INSTALL_HINT
        ) from exc

    gpu_time = cp.asarray(time, dtype=cp.float64)
    tiles = tuple(cp.asarray(tile, dtype=cp.float64) for tile in tile_traces)
    root_tiles = tuple(
        tuple(cp.asarray(row) for row in tile) for tile in tile_root_rows
    )
    singular_tiles = tuple(
        tuple(cp.asarray(row, dtype=cp.float64) for row in tile)
        for tile in tile_singular_value_rows
    )
    if not tiles:
        raise ValueError("tile_traces must contain at least one tile")
    if len(root_tiles) != len(tiles) or len(singular_tiles) != len(tiles):
        raise ValueError(
            "tile roots and singular values must have one entry per tile"
        )
    if gpu_time.ndim != 1:
        raise ValueError("time must be one-dimensional")

    row_counts = []
    flat_roots = []
    flat_singular_values = []
    for tile, roots, singular_values in zip(
        tiles,
        root_tiles,
        singular_tiles,
        strict=True,
    ):
        if tile.ndim != 2:
            raise ValueError("each tile trace block must be two-dimensional")
        if tile.shape[1] != gpu_time.shape[0]:
            raise ValueError(
                "each tile trace block must have one column per time sample"
            )
        row_count = int(tile.shape[0])
        if len(roots) != row_count or len(singular_values) != row_count:
            raise ValueError(
                "tile roots and singular values must match tile row count"
            )
        row_counts.append(row_count)
        flat_roots.extend(roots)
        flat_singular_values.extend(singular_values)

    flat_traces = cp.concatenate(tiles, axis=0)
    filtered, flat_legacy_rows = linear_prediction_batched_legacy_rows_cupy(
        gpu_time,
        flat_traces,
        tuple(flat_roots),
        tuple(flat_singular_values),
        solver=solver,
        window_length=window_length,
        polyorder=polyorder,
    )

    filtered_tiles = []
    legacy_tiles = []
    start = 0
    for row_count in row_counts:
        end = start + row_count
        filtered_tiles.append(filtered[start:end])
        legacy_tiles.append(tuple(flat_legacy_rows[start:end]))
        start = end
    return tuple(filtered_tiles), tuple(legacy_tiles)


def linear_prediction_legacy_rows_from_artifacts_numpy(
    time,
    artifacts,
    decay,
    angular_frequency,
    mode_counts,
    singular_value_rows=None,
) -> tuple[tuple[Any, ...], ...]:
    """Convert batched NumPy P2 artifacts to legacy lpfi row tuples."""

    return _linear_prediction_legacy_rows_from_artifacts(
        np,
        time=time,
        artifacts=artifacts,
        decay=decay,
        angular_frequency=angular_frequency,
        mode_counts=mode_counts,
        singular_value_rows=singular_value_rows,
    )


def linear_prediction_legacy_rows_from_artifacts_cupy(
    time,
    artifacts,
    decay,
    angular_frequency,
    mode_counts,
    singular_value_rows=None,
) -> tuple[tuple[Any, ...], ...]:
    """Convert batched CuPy P2 artifacts to legacy lpfi row tuples."""

    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError(
            "CuPy is required for GPU legacy artifact rows."
            + _CUPY_INSTALL_HINT
        ) from exc

    return _linear_prediction_legacy_rows_from_artifacts(
        cp,
        time=time,
        artifacts=artifacts,
        decay=decay,
        angular_frequency=angular_frequency,
        mode_counts=mode_counts,
        singular_value_rows=singular_value_rows,
    )


def compare_cpu_gpu(
    *,
    samples: int = 96,
    n_components: int = 8,
    run_gpu: bool = True,
    roots_backend: str = "eigvals",
):
    time, trace = synthetic_trace(samples)
    cpu = linear_prediction_numpy(
        time,
        trace,
        n_components,
        roots_backend=roots_backend,
    )
    gpu = (
        linear_prediction_cupy(
            time,
            trace,
            n_components,
            roots_backend=roots_backend,
        )
        if run_gpu
        else None
    )

    if gpu is None:
        return LinearPredictionComparison(
            cpu=cpu,
            gpu=None,
            max_abs_reconstruction_diff=None,
            rms_reconstruction_diff=None,
        )

    diff = cpu.reconstruction - gpu.reconstruction
    return LinearPredictionComparison(
        cpu=cpu,
        gpu=gpu,
        max_abs_reconstruction_diff=float(np.max(np.abs(diff))),
        rms_reconstruction_diff=float(np.sqrt(np.mean(diff**2))),
    )


def model_order_sweep(
    time,
    trace,
    components: tuple[int, ...],
    *,
    roots_backend: str = "eigvals",
    relative_tolerance: float = 0.01,
) -> ModelOrderSweep:
    """Sweep requested component counts for one trace.

    The chosen order is the smallest requested component count whose RMS
    residual is within ``relative_tolerance`` of the best residual observed in
    the sweep. This keeps the command useful for noisy traces where the
    largest model is not automatically the most defensible science choice.
    """

    if not components:
        raise ValueError("components must contain at least one value")
    if relative_tolerance < 0:
        raise ValueError("relative_tolerance must be non-negative")

    entries: list[ModelOrderSweepEntry] = []
    for requested in tuple(dict.fromkeys(int(item) for item in components)):
        result = linear_prediction_numpy(
            time,
            trace,
            requested,
            roots_backend=roots_backend,
        )
        stats = _fit_stats_from_result(
            result,
            trace,
            requested_components=requested,
        )
        singular_values = np.asarray(result.singular_values, dtype=np.float64)
        entries.append(
            ModelOrderSweepEntry(
                components=int(requested),
                selected_model_order=stats.selected_model_order,
                matrix_size=stats.matrix_size,
                root_count=stats.root_count,
                selected_root_count=stats.selected_root_count,
                decaying_root_count=stats.decaying_root_count,
                filtered_root_count=stats.filtered_root_count,
                chi2=stats.chi2,
                rms_residual=stats.rms_residual,
                reconstruction_rms_error=stats.rms_residual,
                singular_value_head=_singular_value_slice(
                    singular_values,
                    head=True,
                ),
                singular_value_tail=_singular_value_slice(
                    singular_values,
                    head=False,
                ),
                singular_value_ratio=_singular_value_ratio(singular_values),
                elapsed_s=float(result.elapsed_s),
            )
        )

    if not entries:
        raise ValueError("components must contain at least one value")
    best_rms = min(entry.rms_residual for entry in entries)
    threshold = best_rms * (1.0 + relative_tolerance)
    qualified_entries = (
        entry for entry in entries if entry.rms_residual <= threshold
    )
    best_entry = min(qualified_entries, key=lambda entry: entry.components)

    return ModelOrderSweep(
        samples=int(len(trace)),
        roots_backend=roots_backend,
        relative_tolerance=float(relative_tolerance),
        best_components=best_entry.components,
        best_selected_model_order=best_entry.selected_model_order,
        best_rms_residual=float(best_entry.rms_residual),
        best_reconstruction_rms_error=float(
            best_entry.reconstruction_rms_error
        ),
        entries=tuple(entries),
    )


def benchmark_linear_prediction_p1_batch(
    *,
    samples: int = 96,
    traces: int = 16,
    n_components: int = 8,
    repeat: int = 3,
    run_gpu: bool = True,
    roots_backend: str = "eigvals",
) -> LinearPredictionP1BatchBenchmark:
    """Benchmark serial P1 against a batched CuPy P1 prototype.

    P1 is the SVD + companion-eigenvalue phase that dominates the reference
    fit path. The production app still runs one detector row at a time; this
    benchmark tests whether the expensive fixed-shape part can be batched.
    """

    if repeat < 1:
        raise ValueError("repeat must be positive")
    time, trace_rows = synthetic_trace_batch(samples=samples, traces=traces)
    fit_stats = _fit_stats_for_trace(
        time,
        trace_rows[0],
        n_components=n_components,
        roots_backend=roots_backend,
    )

    cpu_best, cpu_result = _time_best(
        lambda: [
            _linear_prediction_p1_impl(
                xp=np,
                time=time,
                trace=trace,
                n_components=n_components,
                roots_backend=roots_backend,
            )
            for trace in trace_rows
        ],
        repeat=repeat,
    )

    gpu_serial_best = None
    gpu_batched_best = None
    gpu_batch_speedup = None
    max_abs_coefficient_diff = None
    max_abs_eigenvalue_diff = None
    gpu_error = None
    serial_result = None
    batched_result = None

    if run_gpu:
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA devices visible")

            gpu_time = cp.asarray(time, dtype=cp.float64)
            gpu_traces = cp.asarray(trace_rows, dtype=cp.float64)

            _linear_prediction_p1_cupy_serial(
                gpu_time,
                gpu_traces,
                n_components,
                roots_backend=roots_backend,
            )
            if roots_backend == "eigvals":
                _linear_prediction_p1_cupy_batched(
                    gpu_time,
                    gpu_traces,
                    n_components,
                )
            cp.cuda.Stream.null.synchronize()

            gpu_serial_best, serial_result = _time_best(
                lambda: _linear_prediction_p1_cupy_serial(
                    gpu_time,
                    gpu_traces,
                    n_components,
                    roots_backend=roots_backend,
                ),
                repeat=repeat,
                sync=cp.cuda.Stream.null.synchronize,
            )
            if roots_backend == "eigvals":
                gpu_batched_best, batched_result = _time_best(
                    lambda: _linear_prediction_p1_cupy_batched(
                        gpu_time,
                        gpu_traces,
                        n_components,
                    ),
                    repeat=repeat,
                    sync=cp.cuda.Stream.null.synchronize,
                )
                gpu_batch_speedup = gpu_serial_best / gpu_batched_best

            cpu_coefficients = np.stack(
                [item.coefficients for item in cpu_result]
            )
            cpu_eigenvalues = np.stack(
                [_sort_complex(item.eigenvalues) for item in cpu_result]
            )
            coefficient_result = (
                batched_result
                if batched_result is not None
                else serial_result
            )
            if coefficient_result is not None:
                gpu_coefficients = cp.asnumpy(coefficient_result.coefficients)
                max_abs_coefficient_diff = float(
                    np.max(np.abs(cpu_coefficients - gpu_coefficients))
                )
            if roots_backend == "eigvals":
                gpu_eigenvalues = np.stack(
                    [
                        _sort_complex(row)
                        for row in cp.asnumpy(batched_result.eigenvalues)
                    ]
                )
                max_abs_eigenvalue_diff = float(
                    np.max(np.abs(cpu_eigenvalues - gpu_eigenvalues))
                )
        except Exception as exc:
            # GPU availability and library support are environment dependent.
            gpu_error = str(exc)

    return LinearPredictionP1BatchBenchmark(
        samples=int(samples),
        traces=int(traces),
        components=int(n_components),
        repeat=int(repeat),
        fit=fit_stats,
        cpu_serial_best_s=float(cpu_best),
        gpu_serial_best_s=(
            None if gpu_serial_best is None else float(gpu_serial_best)
        ),
        gpu_batched_best_s=(
            None if gpu_batched_best is None else float(gpu_batched_best)
        ),
        gpu_batch_speedup=(
            None if gpu_batch_speedup is None else float(gpu_batch_speedup)
        ),
        max_abs_coefficient_diff=(
            None
            if max_abs_coefficient_diff is None
            else float(max_abs_coefficient_diff)
        ),
        max_abs_eigenvalue_diff=(
            None
            if max_abs_eigenvalue_diff is None
            else float(max_abs_eigenvalue_diff)
        ),
        gpu_error=gpu_error,
    )


def benchmark_linear_prediction_p2(
    *,
    samples: int = 96,
    traces: int = 16,
    n_components: int = 8,
    repeat: int = 3,
    run_gpu: bool = True,
) -> LinearPredictionP2Benchmark:
    """Benchmark fixed-shape P2 least-squares/reconstruction work.

    P2 uses the decays and frequencies selected by the current fit and then
    solves a small least-squares problem per detector row. This benchmark
    keeps those modes fixed across a synthetic row batch so a batched
    implementation can be compared without changing model-order or
    root-selection behavior.
    """

    if repeat < 1:
        raise ValueError("repeat must be positive")
    time, trace_rows = synthetic_trace_batch(samples=samples, traces=traces)
    reference = linear_prediction_numpy(
        time,
        trace_rows[0],
        n_components=n_components,
    )
    fit_stats = _fit_stats_from_result(
        reference,
        trace_rows[0],
        requested_components=n_components,
    )
    decay = np.asarray(reference.decay, dtype=np.float64)
    angular_frequency = np.asarray(
        reference.angular_frequency,
        dtype=np.float64,
    )
    modes = int(angular_frequency.shape[0])

    cpu_best, cpu_result = _time_best(
        lambda: [
            _linear_prediction_p2_impl(
                xp=np,
                time=time,
                trace=trace,
                decay=decay,
                angular_frequency=angular_frequency,
            )
            for trace in trace_rows
        ],
        repeat=repeat,
    )

    gpu_serial_best = None
    gpu_batched_best = None
    gpu_batch_speedup = None
    max_abs_reconstruction_diff = None
    rms_reconstruction_diff = None
    gpu_error = None

    if run_gpu:
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA devices visible")

            gpu_time = cp.asarray(time, dtype=cp.float64)
            gpu_traces = cp.asarray(trace_rows, dtype=cp.float64)
            gpu_decay = cp.asarray(decay, dtype=cp.float64)
            gpu_angular_frequency = cp.asarray(
                angular_frequency,
                dtype=cp.float64,
            )

            _linear_prediction_p2_cupy_serial(
                gpu_time,
                gpu_traces,
                gpu_decay,
                gpu_angular_frequency,
            )
            _linear_prediction_p2_cupy_batched(
                gpu_time,
                gpu_traces,
                gpu_decay,
                gpu_angular_frequency,
            )
            cp.cuda.Stream.null.synchronize()

            gpu_serial_best, _serial_result = _time_best(
                lambda: _linear_prediction_p2_cupy_serial(
                    gpu_time,
                    gpu_traces,
                    gpu_decay,
                    gpu_angular_frequency,
                ),
                repeat=repeat,
                sync=cp.cuda.Stream.null.synchronize,
            )
            gpu_batched_best, batched_result = _time_best(
                lambda: _linear_prediction_p2_cupy_batched(
                    gpu_time,
                    gpu_traces,
                    gpu_decay,
                    gpu_angular_frequency,
                ),
                repeat=repeat,
                sync=cp.cuda.Stream.null.synchronize,
            )
            gpu_batch_speedup = gpu_serial_best / gpu_batched_best

            cpu_reconstruction = np.stack(
                [item.reconstruction for item in cpu_result]
            )
            gpu_reconstruction = cp.asnumpy(batched_result.reconstruction)
            diff = cpu_reconstruction - gpu_reconstruction
            max_abs_reconstruction_diff = float(np.max(np.abs(diff)))
            rms_reconstruction_diff = float(
                np.sqrt(np.mean(np.abs(diff) ** 2))
            )
        except Exception as exc:
            # GPU availability and library support are environment dependent.
            gpu_error = str(exc)

    return LinearPredictionP2Benchmark(
        samples=int(samples),
        traces=int(traces),
        components=int(n_components),
        modes=modes,
        design_columns=int(modes * 2 + 1),
        repeat=int(repeat),
        fit=fit_stats,
        cpu_serial_best_s=float(cpu_best),
        gpu_serial_best_s=(
            None if gpu_serial_best is None else float(gpu_serial_best)
        ),
        gpu_batched_best_s=(
            None if gpu_batched_best is None else float(gpu_batched_best)
        ),
        gpu_batch_speedup=(
            None if gpu_batch_speedup is None else float(gpu_batch_speedup)
        ),
        max_abs_reconstruction_diff=(
            None
            if max_abs_reconstruction_diff is None
            else float(max_abs_reconstruction_diff)
        ),
        rms_reconstruction_diff=(
            None
            if rms_reconstruction_diff is None
            else float(rms_reconstruction_diff)
        ),
        gpu_error=gpu_error,
    )


def benchmark_linear_prediction_savgol(
    *,
    samples: int = 96,
    traces: int = 16,
    window_length: int = 11,
    polyorder: int = 3,
    repeat: int = 3,
    run_gpu: bool = True,
) -> LinearPredictionSavgolBenchmark:
    """Benchmark the fixed Savitzky-Golay smoothing stage."""

    _validate_savgol_benchmark_inputs(
        samples=samples,
        window_length=window_length,
        polyorder=polyorder,
        repeat=repeat,
    )

    from scipy.signal import savgol_filter

    _time, trace_rows = synthetic_trace_batch(
        samples=samples,
        traces=traces,
    )
    cpu_best, cpu_filtered = _time_best(
        lambda: savgol_filter(
            trace_rows,
            window_length,
            polyorder,
            axis=1,
        ),
        repeat=repeat,
    )

    gpu_serial_best = None
    gpu_batched_best = None
    gpu_batch_speedup = None
    max_abs_filter_diff = None
    rms_filter_diff = None
    gpu_error = None

    if run_gpu:
        try:
            import cupy as cp
            from cupyx.scipy.signal import savgol_filter as cupy_savgol_filter

            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA devices visible")

            gpu_traces = cp.asarray(trace_rows, dtype=cp.float64)
            _savgol_cupy_serial(
                gpu_traces,
                window_length=window_length,
                polyorder=polyorder,
                savgol_filter=cupy_savgol_filter,
            )
            _savgol_cupy_batched(
                gpu_traces,
                window_length=window_length,
                polyorder=polyorder,
                savgol_filter=cupy_savgol_filter,
            )
            cp.cuda.Stream.null.synchronize()

            gpu_serial_best, _serial_filtered = _time_best(
                lambda: _savgol_cupy_serial(
                    gpu_traces,
                    window_length=window_length,
                    polyorder=polyorder,
                    savgol_filter=cupy_savgol_filter,
                ),
                repeat=repeat,
                sync=cp.cuda.Stream.null.synchronize,
            )
            gpu_batched_best, gpu_filtered = _time_best(
                lambda: _savgol_cupy_batched(
                    gpu_traces,
                    window_length=window_length,
                    polyorder=polyorder,
                    savgol_filter=cupy_savgol_filter,
                ),
                repeat=repeat,
                sync=cp.cuda.Stream.null.synchronize,
            )
            gpu_batch_speedup = gpu_serial_best / gpu_batched_best

            filtered = cp.asnumpy(gpu_filtered)
            diff = np.asarray(cpu_filtered) - filtered
            max_abs_filter_diff = float(np.max(np.abs(diff)))
            rms_filter_diff = float(np.sqrt(np.mean(np.abs(diff) ** 2)))
        except Exception as exc:
            # GPU availability and library support are environment dependent.
            gpu_error = str(exc)

    return LinearPredictionSavgolBenchmark(
        samples=int(samples),
        traces=int(traces),
        window_length=int(window_length),
        polyorder=int(polyorder),
        repeat=int(repeat),
        cpu_serial_best_s=float(cpu_best),
        gpu_serial_best_s=(
            None if gpu_serial_best is None else float(gpu_serial_best)
        ),
        gpu_batched_best_s=(
            None if gpu_batched_best is None else float(gpu_batched_best)
        ),
        gpu_batch_speedup=(
            None if gpu_batch_speedup is None else float(gpu_batch_speedup)
        ),
        max_abs_filter_diff=(
            None
            if max_abs_filter_diff is None
            else float(max_abs_filter_diff)
        ),
        rms_filter_diff=(
            None if rms_filter_diff is None else float(rms_filter_diff)
        ),
        gpu_error=gpu_error,
    )


def benchmark_linear_prediction_fixed_stages(
    *,
    samples: int = 96,
    traces: int = 16,
    n_components: int = 8,
    window_length: int = 11,
    polyorder: int = 3,
    repeat: int = 3,
    run_gpu: bool = True,
) -> LinearPredictionFixedStagesBenchmark:
    """Benchmark grouped Savitzky-Golay smoothing and fixed-mode P2 work."""

    if n_components <= 0:
        raise ValueError("n_components must be positive")
    _validate_savgol_benchmark_inputs(
        samples=samples,
        window_length=window_length,
        polyorder=polyorder,
        repeat=repeat,
    )

    from scipy.signal import savgol_filter

    time, trace_rows = synthetic_trace_batch(samples=samples, traces=traces)
    reference_trace = savgol_filter(
        trace_rows[0],
        window_length,
        polyorder,
    )
    reference = linear_prediction_numpy(
        time,
        reference_trace,
        n_components=n_components,
    )
    fit_stats = _fit_stats_from_result(
        reference,
        reference_trace,
        requested_components=n_components,
    )
    decay = np.asarray(reference.decay, dtype=np.float64)
    angular_frequency = np.asarray(
        reference.angular_frequency,
        dtype=np.float64,
    )
    modes = int(angular_frequency.shape[0])

    cpu_best, cpu_result = _time_best(
        lambda: _fixed_stages_numpy_serial(
            time,
            trace_rows,
            decay,
            angular_frequency,
            window_length=window_length,
            polyorder=polyorder,
            savgol_filter=savgol_filter,
        ),
        repeat=repeat,
    )

    gpu_serial_best = None
    gpu_batched_best = None
    gpu_batch_speedup = None
    max_abs_filter_diff = None
    rms_filter_diff = None
    max_abs_reconstruction_diff = None
    rms_reconstruction_diff = None
    gpu_error = None

    if run_gpu:
        try:
            import cupy as cp
            from cupyx.scipy.signal import savgol_filter as cupy_savgol_filter

            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA devices visible")

            gpu_time = cp.asarray(time, dtype=cp.float64)
            gpu_traces = cp.asarray(trace_rows, dtype=cp.float64)
            gpu_decay = cp.asarray(decay, dtype=cp.float64)
            gpu_angular_frequency = cp.asarray(
                angular_frequency,
                dtype=cp.float64,
            )

            _fixed_stages_cupy_serial(
                gpu_time,
                gpu_traces,
                gpu_decay,
                gpu_angular_frequency,
                window_length=window_length,
                polyorder=polyorder,
                savgol_filter=cupy_savgol_filter,
            )
            _fixed_stages_cupy_batched(
                gpu_time,
                gpu_traces,
                gpu_decay,
                gpu_angular_frequency,
                window_length=window_length,
                polyorder=polyorder,
                savgol_filter=cupy_savgol_filter,
            )
            cp.cuda.Stream.null.synchronize()

            gpu_serial_best, _serial_result = _time_best(
                lambda: _fixed_stages_cupy_serial(
                    gpu_time,
                    gpu_traces,
                    gpu_decay,
                    gpu_angular_frequency,
                    window_length=window_length,
                    polyorder=polyorder,
                    savgol_filter=cupy_savgol_filter,
                ),
                repeat=repeat,
                sync=cp.cuda.Stream.null.synchronize,
            )
            gpu_batched_best, gpu_result = _time_best(
                lambda: _fixed_stages_cupy_batched(
                    gpu_time,
                    gpu_traces,
                    gpu_decay,
                    gpu_angular_frequency,
                    window_length=window_length,
                    polyorder=polyorder,
                    savgol_filter=cupy_savgol_filter,
                ),
                repeat=repeat,
                sync=cp.cuda.Stream.null.synchronize,
            )
            gpu_batch_speedup = gpu_serial_best / gpu_batched_best

            cpu_filtered, cpu_p2 = cpu_result
            gpu_filtered, gpu_p2 = gpu_result
            filtered_diff = np.asarray(cpu_filtered) - cp.asnumpy(
                gpu_filtered
            )
            reconstruction_diff = np.asarray(
                cpu_p2.reconstruction
            ) - cp.asnumpy(gpu_p2.reconstruction)
            max_abs_filter_diff = float(np.max(np.abs(filtered_diff)))
            rms_filter_diff = float(
                np.sqrt(np.mean(np.abs(filtered_diff) ** 2))
            )
            max_abs_reconstruction_diff = float(
                np.max(np.abs(reconstruction_diff))
            )
            rms_reconstruction_diff = float(
                np.sqrt(np.mean(np.abs(reconstruction_diff) ** 2))
            )
        except Exception as exc:
            # GPU availability and library support are environment dependent.
            gpu_error = str(exc)

    return LinearPredictionFixedStagesBenchmark(
        samples=int(samples),
        traces=int(traces),
        components=int(n_components),
        modes=modes,
        design_columns=int(modes * 2 + 1),
        window_length=int(window_length),
        polyorder=int(polyorder),
        repeat=int(repeat),
        fit=fit_stats,
        cpu_serial_best_s=float(cpu_best),
        gpu_serial_best_s=(
            None if gpu_serial_best is None else float(gpu_serial_best)
        ),
        gpu_batched_best_s=(
            None if gpu_batched_best is None else float(gpu_batched_best)
        ),
        gpu_batch_speedup=(
            None if gpu_batch_speedup is None else float(gpu_batch_speedup)
        ),
        max_abs_filter_diff=(
            None
            if max_abs_filter_diff is None
            else float(max_abs_filter_diff)
        ),
        rms_filter_diff=(
            None if rms_filter_diff is None else float(rms_filter_diff)
        ),
        max_abs_reconstruction_diff=(
            None
            if max_abs_reconstruction_diff is None
            else float(max_abs_reconstruction_diff)
        ),
        rms_reconstruction_diff=(
            None
            if rms_reconstruction_diff is None
            else float(rms_reconstruction_diff)
        ),
        gpu_error=gpu_error,
    )


def benchmark_linear_prediction_variable_p2(
    *,
    samples: int = 96,
    traces: int = 16,
    n_components: int = 8,
    window_length: int = 11,
    polyorder: int = 3,
    repeat: int = 3,
    run_gpu: bool = True,
    batched_solver: str = "pinv",
    time=None,
    trace_rows=None,
) -> LinearPredictionVariableP2Benchmark:
    """Benchmark P2 with each row's own fitted mode count and modes."""

    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if batched_solver not in (
        "pinv",
        "grouped-pinv",
        "grouped-normal",
        "grouped-qr",
    ):
        raise ValueError(
            "batched_solver must be 'pinv', 'grouped-pinv', "
            "'grouped-normal', or 'grouped-qr'"
        )
    if (time is None) != (trace_rows is None):
        raise ValueError("time and trace_rows must be provided together")
    if time is None:
        time, trace_rows = synthetic_trace_batch(
            samples=samples,
            traces=traces,
        )
    else:
        time, trace_rows = _validate_trace_batch(time, trace_rows)

    _validate_savgol_benchmark_inputs(
        samples=int(time.shape[0]),
        window_length=window_length,
        polyorder=polyorder,
        repeat=repeat,
    )

    from scipy.signal import savgol_filter

    filtered_rows = savgol_filter(
        trace_rows,
        window_length,
        polyorder,
        axis=1,
    )

    mode_start = perf_counter()
    references = [
        linear_prediction_numpy(
            time,
            filtered,
            n_components=n_components,
        )
        for filtered in filtered_rows
    ]
    mode_reference_elapsed_s = perf_counter() - mode_start
    fit_stats = tuple(
        _fit_stats_from_result(
            reference,
            filtered,
            requested_components=n_components,
        )
        for reference, filtered in zip(references, filtered_rows, strict=True)
    )
    decays = tuple(
        np.asarray(item.decay, dtype=np.float64) for item in references
    )
    angular_frequencies = tuple(
        np.asarray(item.angular_frequency, dtype=np.float64)
        for item in references
    )
    mode_counts = tuple(int(item.shape[0]) for item in angular_frequencies)
    max_modes = max(mode_counts, default=0)
    mode_groups = _linear_prediction_mode_groups(mode_counts)
    max_design_columns = int(max_modes * 2 + 1)
    padded_design_entries = int(len(mode_counts) * max_design_columns)
    grouped_design_entries = int(
        sum(group.trace_count * group.design_columns for group in mode_groups)
    )
    if grouped_design_entries:
        padding_overhead_ratio = (
            padded_design_entries / grouped_design_entries
        )
    else:
        padding_overhead_ratio = 1.0

    cpu_best, cpu_reconstruction = _time_best(
        lambda: _variable_p2_numpy_serial(
            time,
            filtered_rows,
            decays,
            angular_frequencies,
        ),
        repeat=repeat,
    )

    gpu_serial_best = None
    gpu_batched_best = None
    gpu_batch_speedup = None
    max_abs_reconstruction_diff = None
    rms_reconstruction_diff = None
    gpu_error = None

    if run_gpu:
        try:
            import cupy as cp

            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA devices visible")

            decay_padded, angular_frequency_padded = _pad_mode_arrays(
                decays,
                angular_frequencies,
                max_modes=max_modes,
            )
            gpu_time = cp.asarray(time, dtype=cp.float64)
            gpu_traces = cp.asarray(filtered_rows, dtype=cp.float64)
            gpu_decay = cp.asarray(decay_padded, dtype=cp.float64)
            gpu_angular_frequency = cp.asarray(
                angular_frequency_padded,
                dtype=cp.float64,
            )
            gpu_mode_counts = cp.asarray(mode_counts, dtype=cp.int64)

            _variable_p2_cupy_serial(
                gpu_time,
                gpu_traces,
                gpu_decay,
                gpu_angular_frequency,
                tuple(mode_counts),
            )
            _variable_p2_cupy_batched_solver(
                gpu_time,
                gpu_traces,
                gpu_decay,
                gpu_angular_frequency,
                gpu_mode_counts,
                solver=batched_solver,
            )
            cp.cuda.Stream.null.synchronize()

            gpu_serial_best, _serial_reconstruction = _time_best(
                lambda: _variable_p2_cupy_serial(
                    gpu_time,
                    gpu_traces,
                    gpu_decay,
                    gpu_angular_frequency,
                    tuple(mode_counts),
                ),
                repeat=repeat,
                sync=cp.cuda.Stream.null.synchronize,
            )
            gpu_batched_best, gpu_reconstruction = _time_best(
                lambda: _variable_p2_cupy_batched_solver(
                    gpu_time,
                    gpu_traces,
                    gpu_decay,
                    gpu_angular_frequency,
                    gpu_mode_counts,
                    solver=batched_solver,
                ),
                repeat=repeat,
                sync=cp.cuda.Stream.null.synchronize,
            )
            gpu_batch_speedup = gpu_serial_best / gpu_batched_best
            if not bool(cp.all(cp.isfinite(gpu_reconstruction)).get()):
                raise RuntimeError(
                    f"{batched_solver} produced non-finite GPU reconstruction"
                )

            reconstruction_diff = np.asarray(cpu_reconstruction) - cp.asnumpy(
                gpu_reconstruction
            )
            max_abs_reconstruction_diff = float(
                np.max(np.abs(reconstruction_diff))
            )
            rms_reconstruction_diff = float(
                np.sqrt(np.mean(np.abs(reconstruction_diff) ** 2))
            )
        except Exception as exc:
            # GPU availability and library support are environment dependent.
            gpu_error = str(exc)

    return LinearPredictionVariableP2Benchmark(
        samples=int(time.shape[0]),
        traces=int(trace_rows.shape[0]),
        components=int(n_components),
        batched_solver=batched_solver,
        window_length=int(window_length),
        polyorder=int(polyorder),
        repeat=int(repeat),
        fits=fit_stats,
        mode_count_min=int(min(mode_counts, default=0)),
        mode_count_max=int(max_modes),
        mode_count_unique=tuple(sorted(set(mode_counts))),
        mode_groups=mode_groups,
        max_design_columns=max_design_columns,
        padded_design_entries=padded_design_entries,
        grouped_design_entries=grouped_design_entries,
        padding_overhead_ratio=float(padding_overhead_ratio),
        mode_reference_elapsed_s=float(mode_reference_elapsed_s),
        cpu_serial_best_s=float(cpu_best),
        gpu_serial_best_s=(
            None if gpu_serial_best is None else float(gpu_serial_best)
        ),
        gpu_batched_best_s=(
            None if gpu_batched_best is None else float(gpu_batched_best)
        ),
        gpu_batch_speedup=(
            None if gpu_batch_speedup is None else float(gpu_batch_speedup)
        ),
        max_abs_reconstruction_diff=(
            None
            if max_abs_reconstruction_diff is None
            else float(max_abs_reconstruction_diff)
        ),
        rms_reconstruction_diff=(
            None
            if rms_reconstruction_diff is None
            else float(rms_reconstruction_diff)
        ),
        gpu_error=gpu_error,
    )


def benchmark_linear_prediction_variable_stages(
    *,
    samples: int = 96,
    traces: int = 16,
    n_components: int = 8,
    window_length: int = 11,
    polyorder: int = 3,
    repeat: int = 3,
    run_gpu: bool = True,
    batched_solver: str = "grouped-pinv",
    time=None,
    trace_rows=None,
) -> LinearPredictionVariableStagesBenchmark:
    """Benchmark batched smoothing and P2 with per-row fitted modes."""

    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if batched_solver not in (
        "pinv",
        "grouped-pinv",
        "grouped-normal",
        "grouped-qr",
    ):
        raise ValueError(
            "batched_solver must be 'pinv', 'grouped-pinv', "
            "'grouped-normal', or 'grouped-qr'"
        )
    if (time is None) != (trace_rows is None):
        raise ValueError("time and trace_rows must be provided together")
    if time is None:
        time, trace_rows = synthetic_trace_batch(
            samples=samples,
            traces=traces,
        )
    else:
        time, trace_rows = _validate_trace_batch(time, trace_rows)

    _validate_savgol_benchmark_inputs(
        samples=int(time.shape[0]),
        window_length=window_length,
        polyorder=polyorder,
        repeat=repeat,
    )

    from scipy.signal import savgol_filter

    filtered_rows = savgol_filter(
        trace_rows,
        window_length,
        polyorder,
        axis=1,
    )
    mode_start = perf_counter()
    references = [
        linear_prediction_numpy(
            time,
            filtered,
            n_components=n_components,
        )
        for filtered in filtered_rows
    ]
    mode_reference_elapsed_s = perf_counter() - mode_start
    fit_stats = tuple(
        _fit_stats_from_result(
            reference,
            filtered,
            requested_components=n_components,
        )
        for reference, filtered in zip(references, filtered_rows, strict=True)
    )
    decays = tuple(
        np.asarray(item.decay, dtype=np.float64) for item in references
    )
    angular_frequencies = tuple(
        np.asarray(item.angular_frequency, dtype=np.float64)
        for item in references
    )
    mode_counts = tuple(int(item.shape[0]) for item in angular_frequencies)
    max_modes = max(mode_counts, default=0)
    mode_groups = _linear_prediction_mode_groups(mode_counts)
    max_design_columns = int(max_modes * 2 + 1)
    padded_design_entries = int(len(mode_counts) * max_design_columns)
    grouped_design_entries = int(
        sum(group.trace_count * group.design_columns for group in mode_groups)
    )
    if grouped_design_entries:
        padding_overhead_ratio = (
            padded_design_entries / grouped_design_entries
        )
    else:
        padding_overhead_ratio = 1.0

    cpu_best, cpu_result = _time_best(
        lambda: _variable_stages_numpy_serial(
            time,
            trace_rows,
            decays,
            angular_frequencies,
            window_length=window_length,
            polyorder=polyorder,
            savgol_filter=savgol_filter,
        ),
        repeat=repeat,
    )

    gpu_serial_best = None
    gpu_batched_best = None
    gpu_batch_speedup = None
    max_abs_filter_diff = None
    rms_filter_diff = None
    max_abs_reconstruction_diff = None
    rms_reconstruction_diff = None
    gpu_error = None

    if run_gpu:
        try:
            import cupy as cp
            from cupyx.scipy.signal import savgol_filter as cupy_savgol_filter

            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA devices visible")

            decay_padded, angular_frequency_padded = _pad_mode_arrays(
                decays,
                angular_frequencies,
                max_modes=max_modes,
            )
            gpu_time = cp.asarray(time, dtype=cp.float64)
            gpu_traces = cp.asarray(trace_rows, dtype=cp.float64)
            gpu_decay = cp.asarray(decay_padded, dtype=cp.float64)
            gpu_angular_frequency = cp.asarray(
                angular_frequency_padded,
                dtype=cp.float64,
            )
            gpu_mode_counts = cp.asarray(mode_counts, dtype=cp.int64)

            _variable_stages_cupy_serial(
                gpu_time,
                gpu_traces,
                gpu_decay,
                gpu_angular_frequency,
                tuple(mode_counts),
                window_length=window_length,
                polyorder=polyorder,
                savgol_filter=cupy_savgol_filter,
            )
            _variable_stages_cupy_batched(
                gpu_time,
                gpu_traces,
                gpu_decay,
                gpu_angular_frequency,
                gpu_mode_counts,
                solver=batched_solver,
                window_length=window_length,
                polyorder=polyorder,
                savgol_filter=cupy_savgol_filter,
            )
            cp.cuda.Stream.null.synchronize()

            gpu_serial_best, _serial_result = _time_best(
                lambda: _variable_stages_cupy_serial(
                    gpu_time,
                    gpu_traces,
                    gpu_decay,
                    gpu_angular_frequency,
                    tuple(mode_counts),
                    window_length=window_length,
                    polyorder=polyorder,
                    savgol_filter=cupy_savgol_filter,
                ),
                repeat=repeat,
                sync=cp.cuda.Stream.null.synchronize,
            )
            gpu_batched_best, gpu_result = _time_best(
                lambda: _variable_stages_cupy_batched(
                    gpu_time,
                    gpu_traces,
                    gpu_decay,
                    gpu_angular_frequency,
                    gpu_mode_counts,
                    solver=batched_solver,
                    window_length=window_length,
                    polyorder=polyorder,
                    savgol_filter=cupy_savgol_filter,
                ),
                repeat=repeat,
                sync=cp.cuda.Stream.null.synchronize,
            )
            gpu_batch_speedup = gpu_serial_best / gpu_batched_best

            cpu_filtered, cpu_reconstruction = cpu_result
            gpu_filtered, gpu_reconstruction = gpu_result
            if not bool(cp.all(cp.isfinite(gpu_reconstruction)).get()):
                raise RuntimeError(
                    f"{batched_solver} produced non-finite GPU reconstruction"
                )
            filtered_diff = np.asarray(cpu_filtered) - cp.asnumpy(
                gpu_filtered
            )
            reconstruction_diff = np.asarray(cpu_reconstruction) - cp.asnumpy(
                gpu_reconstruction
            )
            max_abs_filter_diff = float(np.max(np.abs(filtered_diff)))
            rms_filter_diff = float(
                np.sqrt(np.mean(np.abs(filtered_diff) ** 2))
            )
            max_abs_reconstruction_diff = float(
                np.max(np.abs(reconstruction_diff))
            )
            rms_reconstruction_diff = float(
                np.sqrt(np.mean(np.abs(reconstruction_diff) ** 2))
            )
        except Exception as exc:
            # GPU availability and library support are environment dependent.
            gpu_error = str(exc)

    return LinearPredictionVariableStagesBenchmark(
        samples=int(time.shape[0]),
        traces=int(trace_rows.shape[0]),
        components=int(n_components),
        batched_solver=batched_solver,
        window_length=int(window_length),
        polyorder=int(polyorder),
        repeat=int(repeat),
        fits=fit_stats,
        mode_count_min=int(min(mode_counts, default=0)),
        mode_count_max=int(max_modes),
        mode_count_unique=tuple(sorted(set(mode_counts))),
        mode_groups=mode_groups,
        max_design_columns=max_design_columns,
        padded_design_entries=padded_design_entries,
        grouped_design_entries=grouped_design_entries,
        padding_overhead_ratio=float(padding_overhead_ratio),
        mode_reference_elapsed_s=float(mode_reference_elapsed_s),
        cpu_serial_best_s=float(cpu_best),
        gpu_serial_best_s=(
            None if gpu_serial_best is None else float(gpu_serial_best)
        ),
        gpu_batched_best_s=(
            None if gpu_batched_best is None else float(gpu_batched_best)
        ),
        gpu_batch_speedup=(
            None if gpu_batch_speedup is None else float(gpu_batch_speedup)
        ),
        max_abs_filter_diff=(
            None
            if max_abs_filter_diff is None
            else float(max_abs_filter_diff)
        ),
        rms_filter_diff=(
            None if rms_filter_diff is None else float(rms_filter_diff)
        ),
        max_abs_reconstruction_diff=(
            None
            if max_abs_reconstruction_diff is None
            else float(max_abs_reconstruction_diff)
        ),
        rms_reconstruction_diff=(
            None
            if rms_reconstruction_diff is None
            else float(rms_reconstruction_diff)
        ),
        gpu_error=gpu_error,
    )


def benchmark_linear_prediction_variable_artifacts(
    *,
    samples: int = 96,
    traces: int = 16,
    n_components: int = 8,
    window_length: int = 11,
    polyorder: int = 3,
    repeat: int = 3,
    run_gpu: bool = True,
    batched_solver: str = "grouped-pinv",
    time=None,
    trace_rows=None,
) -> LinearPredictionVariableArtifactsBenchmark:
    """Benchmark full P2 artifacts with each row's fitted modes.

    This is a production-parity harness for the artifacts consumed by the
    reference wrapper after P2: coefficients, amplitudes, phases,
    time-domain components, reconstruction, spectra, frequency centers, and
    chi2. It preserves the current P1 root/model-order selection and only
    tests whether the post-root P2 work can be batched without changing those
    artifacts.
    """

    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if batched_solver not in ("pinv", "grouped-pinv"):
        raise ValueError("batched_solver must be 'pinv' or 'grouped-pinv'")
    if (time is None) != (trace_rows is None):
        raise ValueError("time and trace_rows must be provided together")
    if time is None:
        time, trace_rows = synthetic_trace_batch(
            samples=samples,
            traces=traces,
        )
    else:
        time, trace_rows = _validate_trace_batch(time, trace_rows)

    _validate_savgol_benchmark_inputs(
        samples=int(time.shape[0]),
        window_length=window_length,
        polyorder=polyorder,
        repeat=repeat,
    )

    from scipy.signal import savgol_filter

    filtered_rows = savgol_filter(
        trace_rows,
        window_length,
        polyorder,
        axis=1,
    )
    mode_start = perf_counter()
    references = [
        linear_prediction_numpy(
            time,
            filtered,
            n_components=n_components,
        )
        for filtered in filtered_rows
    ]
    mode_reference_elapsed_s = perf_counter() - mode_start
    fit_stats = tuple(
        _fit_stats_from_result(
            reference,
            filtered,
            requested_components=n_components,
        )
        for reference, filtered in zip(references, filtered_rows, strict=True)
    )
    decays = tuple(
        np.asarray(item.decay, dtype=np.float64) for item in references
    )
    angular_frequencies = tuple(
        np.asarray(item.angular_frequency, dtype=np.float64)
        for item in references
    )
    mode_counts = tuple(int(item.shape[0]) for item in angular_frequencies)
    max_modes = max(mode_counts, default=0)
    mode_groups = _linear_prediction_mode_groups(mode_counts)

    cpu_best, cpu_result = _time_best(
        lambda: _variable_artifacts_numpy_serial(
            time,
            trace_rows,
            decays,
            angular_frequencies,
            max_modes=max_modes,
            window_length=window_length,
            polyorder=polyorder,
            savgol_filter=savgol_filter,
        ),
        repeat=repeat,
    )

    gpu_serial_best = None
    gpu_batched_best = None
    gpu_batch_speedup = None
    max_abs_filter_diff = None
    rms_filter_diff = None
    max_abs_coefficient_diff = None
    max_abs_amplitude_diff = None
    max_abs_phase_diff = None
    max_abs_frequency_center_diff = None
    max_abs_time_component_diff = None
    max_abs_reconstruction_diff = None
    rms_reconstruction_diff = None
    max_abs_spectrum_component_diff = None
    max_abs_spectrum_total_diff = None
    max_abs_chi2_diff = None
    gpu_error = None

    if run_gpu:
        try:
            import cupy as cp
            from cupyx.scipy.signal import savgol_filter as cupy_savgol_filter

            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA devices visible")

            decay_padded, angular_frequency_padded = _pad_mode_arrays(
                decays,
                angular_frequencies,
                max_modes=max_modes,
            )
            gpu_time = cp.asarray(time, dtype=cp.float64)
            gpu_traces = cp.asarray(trace_rows, dtype=cp.float64)
            gpu_decay = cp.asarray(decay_padded, dtype=cp.float64)
            gpu_angular_frequency = cp.asarray(
                angular_frequency_padded,
                dtype=cp.float64,
            )
            gpu_mode_counts = cp.asarray(mode_counts, dtype=cp.int64)

            _variable_artifacts_cupy_serial(
                gpu_time,
                gpu_traces,
                gpu_decay,
                gpu_angular_frequency,
                tuple(mode_counts),
                max_modes=max_modes,
                window_length=window_length,
                polyorder=polyorder,
                savgol_filter=cupy_savgol_filter,
            )
            _variable_artifacts_cupy_batched(
                gpu_time,
                gpu_traces,
                gpu_decay,
                gpu_angular_frequency,
                gpu_mode_counts,
                solver=batched_solver,
                window_length=window_length,
                polyorder=polyorder,
                savgol_filter=cupy_savgol_filter,
            )
            cp.cuda.Stream.null.synchronize()

            gpu_serial_best, _serial_result = _time_best(
                lambda: _variable_artifacts_cupy_serial(
                    gpu_time,
                    gpu_traces,
                    gpu_decay,
                    gpu_angular_frequency,
                    tuple(mode_counts),
                    max_modes=max_modes,
                    window_length=window_length,
                    polyorder=polyorder,
                    savgol_filter=cupy_savgol_filter,
                ),
                repeat=repeat,
                sync=cp.cuda.Stream.null.synchronize,
            )
            gpu_batched_best, gpu_result = _time_best(
                lambda: _variable_artifacts_cupy_batched(
                    gpu_time,
                    gpu_traces,
                    gpu_decay,
                    gpu_angular_frequency,
                    gpu_mode_counts,
                    solver=batched_solver,
                    window_length=window_length,
                    polyorder=polyorder,
                    savgol_filter=cupy_savgol_filter,
                ),
                repeat=repeat,
                sync=cp.cuda.Stream.null.synchronize,
            )
            gpu_batch_speedup = gpu_serial_best / gpu_batched_best

            cpu_filtered, cpu_artifacts = cpu_result
            gpu_filtered, gpu_artifacts = gpu_result
            finite_reconstruction = cp.all(
                cp.isfinite(gpu_artifacts.reconstruction)
            )
            if not bool(finite_reconstruction.get()):
                raise RuntimeError(
                    f"{batched_solver} produced non-finite GPU reconstruction"
                )
            gpu_artifacts_np = _p2_artifacts_to_numpy(cp, gpu_artifacts)
            max_abs_filter_diff = _max_abs_diff(
                np.asarray(cpu_filtered),
                cp.asnumpy(gpu_filtered),
            )
            rms_filter_diff = _rms_diff(
                np.asarray(cpu_filtered),
                cp.asnumpy(gpu_filtered),
            )
            max_abs_coefficient_diff = _max_abs_diff(
                cpu_artifacts.coefficients,
                gpu_artifacts_np.coefficients,
            )
            max_abs_amplitude_diff = _max_abs_diff(
                cpu_artifacts.amplitude,
                gpu_artifacts_np.amplitude,
            )
            max_abs_phase_diff = _max_abs_diff(
                cpu_artifacts.phase,
                gpu_artifacts_np.phase,
            )
            max_abs_frequency_center_diff = _max_abs_diff(
                cpu_artifacts.frequency_centers,
                gpu_artifacts_np.frequency_centers,
            )
            max_abs_time_component_diff = _max_abs_diff(
                cpu_artifacts.time_components,
                gpu_artifacts_np.time_components,
            )
            max_abs_reconstruction_diff = _max_abs_diff(
                cpu_artifacts.reconstruction,
                gpu_artifacts_np.reconstruction,
            )
            rms_reconstruction_diff = _rms_diff(
                cpu_artifacts.reconstruction,
                gpu_artifacts_np.reconstruction,
            )
            max_abs_spectrum_component_diff = _max_abs_diff(
                cpu_artifacts.spectrum_components,
                gpu_artifacts_np.spectrum_components,
            )
            max_abs_spectrum_total_diff = _max_abs_diff(
                cpu_artifacts.spectrum_total,
                gpu_artifacts_np.spectrum_total,
            )
            max_abs_chi2_diff = _max_abs_diff(
                cpu_artifacts.chi2,
                gpu_artifacts_np.chi2,
            )
        except Exception as exc:
            # GPU availability and library support are environment dependent.
            gpu_error = str(exc)

    return LinearPredictionVariableArtifactsBenchmark(
        samples=int(time.shape[0]),
        traces=int(trace_rows.shape[0]),
        components=int(n_components),
        batched_solver=batched_solver,
        window_length=int(window_length),
        polyorder=int(polyorder),
        repeat=int(repeat),
        fits=fit_stats,
        mode_count_min=int(min(mode_counts, default=0)),
        mode_count_max=int(max_modes),
        mode_count_unique=tuple(sorted(set(mode_counts))),
        mode_groups=mode_groups,
        max_design_columns=int(max_modes * 2 + 1),
        mode_reference_elapsed_s=float(mode_reference_elapsed_s),
        cpu_serial_best_s=float(cpu_best),
        gpu_serial_best_s=(
            None if gpu_serial_best is None else float(gpu_serial_best)
        ),
        gpu_batched_best_s=(
            None if gpu_batched_best is None else float(gpu_batched_best)
        ),
        gpu_batch_speedup=(
            None if gpu_batch_speedup is None else float(gpu_batch_speedup)
        ),
        max_abs_filter_diff=(
            None
            if max_abs_filter_diff is None
            else float(max_abs_filter_diff)
        ),
        rms_filter_diff=(
            None if rms_filter_diff is None else float(rms_filter_diff)
        ),
        max_abs_coefficient_diff=(
            None
            if max_abs_coefficient_diff is None
            else float(max_abs_coefficient_diff)
        ),
        max_abs_amplitude_diff=(
            None
            if max_abs_amplitude_diff is None
            else float(max_abs_amplitude_diff)
        ),
        max_abs_phase_diff=(
            None if max_abs_phase_diff is None else float(max_abs_phase_diff)
        ),
        max_abs_frequency_center_diff=(
            None
            if max_abs_frequency_center_diff is None
            else float(max_abs_frequency_center_diff)
        ),
        max_abs_time_component_diff=(
            None
            if max_abs_time_component_diff is None
            else float(max_abs_time_component_diff)
        ),
        max_abs_reconstruction_diff=(
            None
            if max_abs_reconstruction_diff is None
            else float(max_abs_reconstruction_diff)
        ),
        rms_reconstruction_diff=(
            None
            if rms_reconstruction_diff is None
            else float(rms_reconstruction_diff)
        ),
        max_abs_spectrum_component_diff=(
            None
            if max_abs_spectrum_component_diff is None
            else float(max_abs_spectrum_component_diff)
        ),
        max_abs_spectrum_total_diff=(
            None
            if max_abs_spectrum_total_diff is None
            else float(max_abs_spectrum_total_diff)
        ),
        max_abs_chi2_diff=(
            None if max_abs_chi2_diff is None else float(max_abs_chi2_diff)
        ),
        gpu_error=gpu_error,
    )


def benchmark_linear_prediction_runtime_bridge(
    *,
    samples: int = 96,
    tiles: int = 4,
    rows_per_tile: int = 16,
    n_components: int = 8,
    window_length: int = 11,
    polyorder: int = 3,
    repeat: int = 3,
    run_gpu: bool = True,
    batched_solver: str = "grouped-pinv",
    time=None,
    trace_rows=None,
) -> LinearPredictionRuntimeBridgeBenchmark:
    """Compare post-P1 serial, per-tile batched, and multi-tile batched work.

    The P1 roots and singular values are precomputed once and reused by every
    measured path. That keeps root/model-order behavior fixed while timing the
    runtime bridge shapes that sit after P1 in the production wrapper.
    """

    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if tiles <= 0:
        raise ValueError("tiles must be positive")
    if rows_per_tile <= 0:
        raise ValueError("rows_per_tile must be positive")
    if batched_solver not in ("pinv", "grouped-pinv"):
        raise ValueError("batched_solver must be 'pinv' or 'grouped-pinv'")
    if (time is None) != (trace_rows is None):
        raise ValueError("time and trace_rows must be provided together")
    if time is None:
        time, trace_rows = synthetic_trace_batch(
            samples=samples,
            traces=tiles * rows_per_tile,
        )
    else:
        time, trace_rows = _validate_trace_batch(time, trace_rows)
        if trace_rows.shape[0] % rows_per_tile:
            raise ValueError(
                "trace row count must be divisible by rows_per_tile"
            )
        tiles = int(trace_rows.shape[0] // rows_per_tile)

    _validate_savgol_benchmark_inputs(
        samples=int(time.shape[0]),
        window_length=window_length,
        polyorder=polyorder,
        repeat=repeat,
    )

    from scipy.signal import savgol_filter

    filtered_rows = savgol_filter(
        trace_rows,
        window_length,
        polyorder,
        axis=1,
    )
    p1_start = perf_counter()
    references = [
        linear_prediction_numpy(
            time,
            filtered,
            n_components=n_components,
        )
        for filtered in filtered_rows
    ]
    p1_rows = [
        _linear_prediction_p1_impl(
            xp=np,
            time=time,
            trace=filtered,
            n_components=n_components,
        )
        for filtered in filtered_rows
    ]
    p1_reference_elapsed_s = perf_counter() - p1_start
    fits = tuple(
        _fit_stats_from_result(
            reference,
            filtered,
            requested_components=n_components,
        )
        for reference, filtered in zip(references, filtered_rows, strict=True)
    )
    tile_traces = tuple(
        trace_rows[start : start + rows_per_tile]
        for start in range(0, trace_rows.shape[0], rows_per_tile)
    )
    tile_roots = tuple(
        tuple(
            item.eigenvalues
            for item in p1_rows[start : start + rows_per_tile]
        )
        for start in range(0, len(p1_rows), rows_per_tile)
    )
    tile_singular_values = tuple(
        tuple(
            item.singular_values
            for item in p1_rows[start : start + rows_per_tile]
        )
        for start in range(0, len(p1_rows), rows_per_tile)
    )
    mode_counts = tuple(int(item.decay.shape[0]) for item in references)
    max_modes = max(mode_counts, default=0)

    gpu_serial_best = None
    gpu_row_batched_best = None
    gpu_multi_tile_best = None
    row_batched_speedup = None
    multi_tile_speedup = None
    multi_vs_row_batched_speedup = None
    max_abs_frequency_center_diff = None
    max_abs_time_component_diff = None
    max_abs_reconstruction_diff = None
    max_abs_amplitude_diff = None
    max_abs_phase_diff = None
    max_abs_chi2_diff = None
    gpu_error = None

    if run_gpu:
        try:
            import cupy as cp
            from cupyx.scipy.signal import savgol_filter as cupy_savgol_filter

            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA devices visible")

            gpu_time = cp.asarray(time, dtype=cp.float64)
            gpu_tile_traces = tuple(
                cp.asarray(tile, dtype=cp.float64) for tile in tile_traces
            )
            gpu_tile_roots = tuple(
                tuple(cp.asarray(row) for row in tile) for tile in tile_roots
            )
            gpu_tile_singular_values = tuple(
                tuple(cp.asarray(row, dtype=cp.float64) for row in tile)
                for tile in tile_singular_values
            )

            _legacy_tiles_cupy_serial(
                gpu_time,
                gpu_tile_traces,
                gpu_tile_roots,
                gpu_tile_singular_values,
                window_length=window_length,
                polyorder=polyorder,
                savgol_filter=cupy_savgol_filter,
            )
            _legacy_tiles_cupy_row_batched(
                gpu_time,
                gpu_tile_traces,
                gpu_tile_roots,
                gpu_tile_singular_values,
                solver=batched_solver,
                window_length=window_length,
                polyorder=polyorder,
            )
            linear_prediction_batched_legacy_tiles_cupy(
                gpu_time,
                gpu_tile_traces,
                gpu_tile_roots,
                gpu_tile_singular_values,
                solver=batched_solver,
                window_length=window_length,
                polyorder=polyorder,
            )
            cp.cuda.Stream.null.synchronize()

            gpu_serial_best, serial_result = _time_best(
                lambda: _legacy_tiles_cupy_serial(
                    gpu_time,
                    gpu_tile_traces,
                    gpu_tile_roots,
                    gpu_tile_singular_values,
                    window_length=window_length,
                    polyorder=polyorder,
                    savgol_filter=cupy_savgol_filter,
                ),
                repeat=repeat,
                sync=cp.cuda.Stream.null.synchronize,
            )
            gpu_row_batched_best, row_batched_result = _time_best(
                lambda: _legacy_tiles_cupy_row_batched(
                    gpu_time,
                    gpu_tile_traces,
                    gpu_tile_roots,
                    gpu_tile_singular_values,
                    solver=batched_solver,
                    window_length=window_length,
                    polyorder=polyorder,
                ),
                repeat=repeat,
                sync=cp.cuda.Stream.null.synchronize,
            )
            gpu_multi_tile_best, multi_tile_result = _time_best(
                lambda: linear_prediction_batched_legacy_tiles_cupy(
                    gpu_time,
                    gpu_tile_traces,
                    gpu_tile_roots,
                    gpu_tile_singular_values,
                    solver=batched_solver,
                    window_length=window_length,
                    polyorder=polyorder,
                ),
                repeat=repeat,
                sync=cp.cuda.Stream.null.synchronize,
            )
            row_batched_speedup = gpu_serial_best / gpu_row_batched_best
            multi_tile_speedup = gpu_serial_best / gpu_multi_tile_best
            multi_vs_row_batched_speedup = (
                gpu_row_batched_best / gpu_multi_tile_best
            )

            _serial_filtered, serial_tiles = serial_result
            _multi_filtered, multi_tiles = multi_tile_result
            _row_filtered, row_tiles = row_batched_result
            _assert_legacy_tiles_close(cp, row_tiles, multi_tiles)
            max_abs_frequency_center_diff = _legacy_tiles_max_abs_diff(
                cp, serial_tiles, multi_tiles, slot=0
            )
            max_abs_time_component_diff = _legacy_tiles_max_abs_diff(
                cp, serial_tiles, multi_tiles, slot=2
            )
            max_abs_reconstruction_diff = _legacy_tiles_max_abs_diff(
                cp, serial_tiles, multi_tiles, slot=3
            )
            max_abs_amplitude_diff = _legacy_tiles_max_abs_diff(
                cp, serial_tiles, multi_tiles, slot=9
            )
            max_abs_phase_diff = _legacy_tiles_max_abs_diff(
                cp, serial_tiles, multi_tiles, slot=10
            )
            max_abs_chi2_diff = _legacy_tiles_max_abs_diff(
                cp, serial_tiles, multi_tiles, slot=11
            )
        except Exception as exc:
            gpu_error = str(exc)

    return LinearPredictionRuntimeBridgeBenchmark(
        samples=int(time.shape[0]),
        tiles=int(tiles),
        rows_per_tile=int(rows_per_tile),
        traces=int(trace_rows.shape[0]),
        components=int(n_components),
        batched_solver=batched_solver,
        window_length=int(window_length),
        polyorder=int(polyorder),
        repeat=int(repeat),
        fits=fits,
        mode_count_min=int(min(mode_counts, default=0)),
        mode_count_max=int(max_modes),
        mode_count_unique=tuple(sorted(set(mode_counts))),
        mode_groups=_linear_prediction_mode_groups(mode_counts),
        max_design_columns=int(max_modes * 2 + 1),
        p1_reference_elapsed_s=float(p1_reference_elapsed_s),
        gpu_serial_per_tile_best_s=(
            None if gpu_serial_best is None else float(gpu_serial_best)
        ),
        gpu_row_batched_per_tile_best_s=(
            None
            if gpu_row_batched_best is None
            else float(gpu_row_batched_best)
        ),
        gpu_multi_tile_grouped_best_s=(
            None
            if gpu_multi_tile_best is None
            else float(gpu_multi_tile_best)
        ),
        row_batched_speedup=(
            None
            if row_batched_speedup is None
            else float(row_batched_speedup)
        ),
        multi_tile_speedup=(
            None if multi_tile_speedup is None else float(multi_tile_speedup)
        ),
        multi_vs_row_batched_speedup=(
            None
            if multi_vs_row_batched_speedup is None
            else float(multi_vs_row_batched_speedup)
        ),
        max_abs_frequency_center_diff=(
            None
            if max_abs_frequency_center_diff is None
            else float(max_abs_frequency_center_diff)
        ),
        max_abs_time_component_diff=(
            None
            if max_abs_time_component_diff is None
            else float(max_abs_time_component_diff)
        ),
        max_abs_reconstruction_diff=(
            None
            if max_abs_reconstruction_diff is None
            else float(max_abs_reconstruction_diff)
        ),
        max_abs_amplitude_diff=(
            None
            if max_abs_amplitude_diff is None
            else float(max_abs_amplitude_diff)
        ),
        max_abs_phase_diff=(
            None if max_abs_phase_diff is None else float(max_abs_phase_diff)
        ),
        max_abs_chi2_diff=(
            None if max_abs_chi2_diff is None else float(max_abs_chi2_diff)
        ),
        gpu_error=gpu_error,
    )


def benchmark_prediction_roots(
    *,
    samples: int = 800,
    traces: int = 16,
    n_components: int = 30,
    repeat: int = 3,
    backends: tuple[str, ...] | None = None,
    run_gpu: bool = True,
) -> PredictionRootsBenchmark:
    """Benchmark root-solving backends on production-shaped coefficients."""

    if repeat < 1:
        raise ValueError("repeat must be positive")
    coefficients = synthetic_prediction_coefficients(
        samples=samples,
        traces=traces,
        n_components=n_components,
    )
    time, trace_rows = synthetic_trace_batch(samples=samples, traces=traces)
    fit_stats = _fit_stats_for_trace(
        time,
        trace_rows[0],
        n_components=n_components,
        roots_backend="eigvals",
    )
    if backends is None:
        backends = (
            "numpy-eigvals",
            "numpy-roots",
            "cupy-eigvals-serial",
            "cupy-eigvals-batched",
            "cupy-roots-serial",
        )

    baseline_roots = _roots_numpy_serial(coefficients, backend="eigvals")
    results: list[PredictionRootsBackendBenchmark] = []
    for backend_name in backends:
        if backend_name.startswith("cupy-") and not run_gpu:
            results.append(
                _skipped_roots_backend(
                    backend_name,
                    coefficients,
                    error="GPU backends skipped",
                )
            )
            continue
        results.append(
            _benchmark_prediction_roots_backend(
                coefficients,
                backend_name=backend_name,
                baseline_roots=baseline_roots,
                repeat=repeat,
            )
        )

    return PredictionRootsBenchmark(
        samples=int(samples),
        traces=int(traces),
        components=int(n_components),
        repeat=int(repeat),
        fit=fit_stats,
        backends=tuple(results),
    )


def _linear_prediction_legacy_rows_from_artifacts(
    xp,
    *,
    time,
    artifacts,
    decay,
    angular_frequency,
    mode_counts,
    singular_value_rows,
) -> tuple[tuple[Any, ...], ...]:
    fit_time = _fit_time_from_bins(xp, time)
    decay = xp.asarray(decay, dtype=xp.float64)
    angular_frequency = xp.asarray(angular_frequency, dtype=xp.float64)
    mode_counts = xp.asarray(mode_counts, dtype=xp.int64)
    frequency_centers = xp.asarray(artifacts.frequency_centers)
    reconstruction = xp.asarray(artifacts.reconstruction)
    frequency = xp.asarray(artifacts.frequency)
    spectrum_components = xp.asarray(artifacts.spectrum_components)
    spectrum_total = xp.asarray(artifacts.spectrum_total)
    amplitude = xp.asarray(artifacts.amplitude)
    phase = xp.asarray(artifacts.phase)
    chi2 = xp.asarray(artifacts.chi2)
    time_components = xp.asarray(artifacts.time_components)

    if decay.shape != angular_frequency.shape:
        raise ValueError("decay and angular_frequency must have same shape")
    if decay.ndim != 2:
        raise ValueError("mode arrays must be two-dimensional")
    row_count = int(decay.shape[0])
    sample_count = int(fit_time.shape[0])
    max_modes = int(decay.shape[1])
    if mode_counts.shape != (row_count,):
        raise ValueError("mode_counts must have one item per row")
    _validate_legacy_artifact_shapes(
        row_count=row_count,
        sample_count=sample_count,
        max_modes=max_modes,
        frequency_centers=frequency_centers,
        reconstruction=reconstruction,
        frequency=frequency,
        spectrum_components=spectrum_components,
        spectrum_total=spectrum_total,
        amplitude=amplitude,
        phase=phase,
        chi2=chi2,
        time_components=time_components,
    )

    if singular_value_rows is None:
        singular_values = tuple(
            xp.zeros(0, dtype=xp.float64) for _row in range(row_count)
        )
    else:
        singular_values = tuple(
            xp.asarray(row, dtype=xp.float64) for row in singular_value_rows
        )
        if len(singular_values) != row_count:
            raise ValueError("singular_value_rows must have one item per row")

    rows = []
    for row in range(row_count):
        mode_count = _scalar_int(mode_counts[row])
        if mode_count < 0 or mode_count > max_modes:
            raise ValueError("mode_counts entries must fit padded modes")
        if mode_count:
            spectrum_1 = spectrum_components[row, :, :mode_count]
        else:
            spectrum_1 = frequency[row, :, None]
        rows.append(
            (
                frequency_centers[row, :mode_count],
                fit_time,
                time_components[row, :, :mode_count],
                reconstruction[row],
                frequency[row],
                spectrum_1,
                spectrum_total[row],
                angular_frequency[row, :mode_count],
                decay[row, :mode_count],
                amplitude[row, :mode_count],
                phase[row, :mode_count],
                chi2[row],
                singular_values[row],
            )
        )
    return tuple(rows)


def _fit_time_from_bins(xp, time):
    bins = xp.asarray(time, dtype=xp.float64)
    if bins.ndim != 1:
        raise ValueError("time must be one-dimensional")
    if bins.shape[0] < 2:
        raise ValueError("time must contain at least two samples")
    n = bins.shape[0] - 1
    return xp.linspace(0, n, n + 1) * (bins[1] - bins[0])


def _validate_legacy_artifact_shapes(
    *,
    row_count: int,
    sample_count: int,
    max_modes: int,
    frequency_centers,
    reconstruction,
    frequency,
    spectrum_components,
    spectrum_total,
    amplitude,
    phase,
    chi2,
    time_components,
):
    mode_shape = (row_count, max_modes)
    for name, value in (
        ("frequency_centers", frequency_centers),
        ("amplitude", amplitude),
        ("phase", phase),
    ):
        if value.shape != mode_shape:
            raise ValueError(f"{name} must have shape (rows, modes)")

    if reconstruction.shape != (row_count, sample_count):
        raise ValueError("reconstruction must have shape (rows, samples)")
    if time_components.shape != (row_count, sample_count, max_modes):
        raise ValueError(
            "time_components must have shape (rows, samples, modes)"
        )
    if frequency.ndim != 2 or int(frequency.shape[0]) != row_count:
        raise ValueError("frequency must have shape (rows, frequencies)")
    frequency_count = int(frequency.shape[1])
    if spectrum_components.shape != (row_count, frequency_count, max_modes):
        raise ValueError(
            "spectrum_components must have shape (rows, frequencies, modes)"
        )
    if spectrum_total.shape != (row_count, frequency_count):
        raise ValueError("spectrum_total must have shape (rows, frequencies)")
    if chi2.shape != (row_count,):
        raise ValueError("chi2 must have one item per row")


def _linear_prediction_mode_batch_from_roots(
    xp,
    *,
    time,
    root_rows,
    singular_value_counts,
) -> LinearPredictionModeBatch:
    root_rows = tuple(root_rows)
    singular_value_counts = tuple(singular_value_counts)
    if len(root_rows) != len(singular_value_counts):
        raise ValueError(
            "root_rows and singular_value_counts must have same length"
        )

    modes = tuple(
        _linear_prediction_modes_from_roots(
            xp,
            time=time,
            roots=roots,
            singular_value_count=_singular_value_count(count),
        )
        for roots, count in zip(
            root_rows,
            singular_value_counts,
            strict=True,
        )
    )
    mode_counts = tuple(int(item.decay.shape[0]) for item in modes)
    max_modes = max(mode_counts, default=0)
    decay_padded, angular_frequency_padded = _pad_mode_arrays(
        tuple(item.decay for item in modes),
        tuple(item.angular_frequency for item in modes),
        max_modes=max_modes,
        xp=xp,
    )
    return LinearPredictionModeBatch(
        decay=decay_padded,
        angular_frequency=angular_frequency_padded,
        mode_counts=xp.asarray(mode_counts, dtype=xp.int64),
        decaying_root_counts=xp.asarray(
            tuple(item.decaying_root_count for item in modes),
            dtype=xp.int64,
        ),
    )


def _singular_value_count(value) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        if shape == ():
            get = getattr(value, "get", None)
            if get is not None:
                return int(get())
            item = getattr(value, "item", None)
            if item is not None:
                return int(item())
        return int(len(value))
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(len(value))


def _linear_prediction_modes_from_roots(
    xp,
    *,
    time,
    roots,
    singular_value_count: int,
) -> LinearPredictionModes:
    bins = xp.asarray(time, dtype=xp.float64)
    roots = xp.asarray(roots)
    if bins.ndim != 1:
        raise ValueError("time must be one-dimensional")
    if bins.shape[0] < 2:
        raise ValueError("time must contain at least two samples")
    if roots.ndim != 1:
        raise ValueError("roots must be one-dimensional")
    if singular_value_count < 0:
        raise ValueError("singular_value_count must be non-negative")

    delta_t = bins[1] - bins[0]
    sorted_roots = xp.sort(roots)[::-1]
    active_roots = sorted_roots[: int(singular_value_count)]
    decay = xp.log(xp.abs(active_roots)) / delta_t
    # XRay's fitted components use exp(-decay * t), so nonnegative
    # fitted decay values are the decaying/stable candidates preserved by
    # the current root filter.
    decaying_root_count = _scalar_int(xp.sum(decay >= 0))
    angular_frequency = xp.angle(active_roots) / delta_t
    order = angular_frequency.argsort()
    angular_frequency = angular_frequency[order]
    decay = decay[order]

    zero_count = _scalar_int(xp.sum(angular_frequency == 0))
    select_count = int((len(angular_frequency) - zero_count) / 2 + zero_count)
    angular_frequency = xp.abs(angular_frequency[:select_count])
    decay = decay[:select_count]

    keep = (angular_frequency >= 0) & (decay >= 0)
    angular_frequency = angular_frequency[keep]
    decay = decay[keep]

    order = (-decay).argsort()
    return LinearPredictionModes(
        decay=decay[order],
        angular_frequency=angular_frequency[order],
        decaying_root_count=decaying_root_count,
    )


def _linear_prediction_impl(
    *,
    xp,
    backend,
    time,
    trace,
    n_components,
    roots_backend,
):
    bins, signal = _validate_inputs(xp, time, trace, n_components)
    nbins = bins.shape[0]
    delta_t = bins[1] - bins[0]
    n = nbins - 1
    m = int(n * 0.75)
    rows = n - m

    if rows <= 0 or m <= 0:
        raise ValueError("trace is too short for linear prediction")

    hankel = xp.zeros((rows, m), dtype=signal.dtype)
    for i in range(rows):
        hankel[i, :] = signal[i + 1 : i + m + 1]

    u, singular_values, vh = xp.linalg.svd(hankel, full_matrices=False)
    selected_model_order = _component_count(
        xp,
        singular_values,
        requested=n_components,
        limit=min(rows, m),
    )
    inv_s = 1 / singular_values[:selected_model_order]
    xvector = signal[:rows]
    prediction_coeff = (
        vh.conj().T[:, :selected_model_order]
        * inv_s
        @ (u.conj().T[:selected_model_order, :] @ xvector)
    )

    roots_result = _prediction_roots_result(
        xp,
        prediction_coeff,
        backend=roots_backend,
    )
    modes = _linear_prediction_modes_from_roots(
        xp,
        time=bins,
        roots=roots_result.roots,
        singular_value_count=len(singular_values),
    )
    decay = modes.decay
    angular_frequency = modes.angular_frequency

    nw = len(angular_frequency)
    fit_time = xp.linspace(0, n, n + 1) * delta_t
    exp_bt = xp.exp(-decay[:, None] * fit_time)
    cos_wt = xp.cos(angular_frequency[:, None] * fit_time)
    sin_wt = xp.sin(angular_frequency[:, None] * fit_time)

    xbar = xp.zeros((n + 1, nw * 2 + 1), dtype=signal.dtype)
    xbar[:, :-1:2] = (exp_bt * cos_wt).T
    xbar[:, 1:-1:2] = (-exp_bt * sin_wt).T
    xbar[:, -1] = 1

    coefficients = xp.linalg.lstsq(xbar, signal, rcond=None)[0]
    a0 = coefficients[: nw * 2 : 2]
    a1 = coefficients[1 : nw * 2 : 2]
    amplitude, phase = _amplitudes_and_phases_backend(xp, a0, a1)

    time_components = (
        amplitude * xp.exp(-decay * fit_time[:, None])
    ) * xp.cos(angular_frequency * fit_time[:, None] + phase)
    reconstruction = xp.sum(time_components, axis=1)
    reconstruction = reconstruction + coefficients[2 * nw]

    max_w = xp.max(angular_frequency) if nw else xp.asarray(1e-5)
    frequency = xp.linspace(0, 1.5 * max_w / (2 * pi), 1000)
    spectrum_components = xp.zeros((len(frequency), nw), dtype=signal.dtype)
    if nw:
        bi = decay / (2 * pi)
        ww = angular_frequency / (2 * pi)
        spectrum_components[:, :nw] = (amplitude * bi)[None, :] / (
            (ww[None, :] - frequency[:, None]) ** 2 + bi[None, :] ** 2
        )
    spectrum_total = xp.sum(spectrum_components, axis=1)
    residual = reconstruction[:-1] - signal[:-1]
    chi2 = xp.sum(residual**2) / (n - 1)

    return LinearPredictionResult(
        backend=backend,
        time=_to_numpy(xp, fit_time),
        time_components=_to_numpy(xp, time_components),
        reconstruction=_to_numpy(xp, reconstruction),
        frequency=_to_numpy(xp, frequency),
        spectrum_components=_to_numpy(xp, spectrum_components),
        spectrum_total=_to_numpy(xp, spectrum_total),
        angular_frequency=_to_numpy(xp, angular_frequency),
        decay=_to_numpy(xp, decay),
        amplitude=_to_numpy(xp, amplitude),
        phase=_to_numpy(xp, phase),
        chi2=float(_to_numpy(xp, chi2)),
        selected_model_order=int(selected_model_order),
        decaying_root_count=modes.decaying_root_count,
        singular_values=_to_numpy(xp, singular_values),
        roots_stats=roots_result.stats,
        elapsed_s=0.0,
    )


def _linear_prediction_p1_coefficients_impl(*, xp, time, trace, n_components):
    bins, signal = _validate_inputs(xp, time, trace, n_components)
    nbins = bins.shape[0]
    n = nbins - 1
    m = int(n * 0.75)
    rows = n - m
    if rows <= 0 or m <= 0:
        raise ValueError("trace is too short for linear prediction")
    hankel = xp.zeros((rows, m), dtype=signal.dtype)
    for i in range(rows):
        hankel[i, :] = signal[i + 1 : i + m + 1]
    u, singular_values, vh = xp.linalg.svd(hankel, full_matrices=False)
    selected_model_order = _component_count(
        xp,
        singular_values,
        requested=n_components,
        limit=min(rows, m),
    )
    inv_s = 1 / singular_values[:selected_model_order]
    xvector = signal[:rows]
    coefficients = (
        vh.conj().T[:, :selected_model_order]
        * inv_s
        @ (u.conj().T[:selected_model_order, :] @ xvector)
    )
    return _P1CoefficientsResult(
        singular_values=singular_values,
        coefficients=coefficients,
        selected_model_order=int(selected_model_order),
    )


def _linear_prediction_p1_impl(
    *,
    xp,
    time,
    trace,
    n_components,
    roots_backend="eigvals",
):
    p1 = _linear_prediction_p1_coefficients_impl(
        xp=xp,
        time=time,
        trace=trace,
        n_components=n_components,
    )
    roots = _prediction_roots_result(
        xp,
        p1.coefficients,
        backend=roots_backend,
    )
    return _P1Result(
        singular_values=p1.singular_values,
        coefficients=p1.coefficients,
        eigenvalues=roots.roots,
        selected_model_order=p1.selected_model_order,
        roots_stats=roots.stats,
    )


def _linear_prediction_p1_cupy_serial(
    time,
    traces,
    n_components,
    *,
    roots_backend="eigvals",
):
    import cupy as cp

    results = [
        _linear_prediction_p1_impl(
            xp=cp,
            time=time,
            trace=trace,
            n_components=n_components,
            roots_backend=roots_backend,
        )
        for trace in traces
    ]
    return _P1Result(
        singular_values=cp.stack([item.singular_values for item in results]),
        coefficients=cp.stack([item.coefficients for item in results]),
        eigenvalues=cp.stack([item.eigenvalues for item in results]),
        selected_model_order=tuple(
            int(item.selected_model_order) for item in results
        ),
    )


def _linear_prediction_p1_cupy_batched(time, traces, n_components):
    import cupy as cp

    bins = cp.asarray(time, dtype=cp.float64)
    signals = cp.asarray(traces, dtype=cp.float64)
    if bins.ndim != 1 or signals.ndim != 2:
        raise ValueError("time must be 1D and traces must be 2D")
    if signals.shape[1] != bins.shape[0]:
        raise ValueError("traces must have one row per time sample")
    if bins.shape[0] < 16:
        raise ValueError("time must contain at least 16 samples")
    if n_components <= 0:
        raise ValueError("n_components must be positive")

    n = bins.shape[0] - 1
    m = int(n * 0.75)
    rows = n - m
    row_index = cp.arange(rows)[:, None]
    col_index = cp.arange(1, m + 1)[None, :]
    hankel = signals[:, row_index + col_index]
    u, singular_values, vh = cp.linalg.svd(hankel, full_matrices=False)
    max_s = cp.max(singular_values, axis=1)
    valid = cp.logical_and(
        cp.isfinite(singular_values),
        singular_values > max_s[:, None] * 1e-12,
    )
    valid_components = int(cp.min(cp.count_nonzero(valid, axis=1)).get())
    selected_model_order = min(int(n_components), rows, m, valid_components)
    if selected_model_order <= 0:
        raise ValueError("insufficient finite singular values")

    v = cp.swapaxes(vh.conj(), -2, -1)[:, :, :selected_model_order]
    u_h = cp.swapaxes(u.conj(), -2, -1)[:, :selected_model_order, :]
    xvector = signals[:, :rows]
    projected = cp.matmul(u_h, xvector[:, :, None])[:, :, 0]
    scaled = projected / singular_values[:, :selected_model_order]
    coefficients = cp.matmul(v, scaled[:, :, None])[:, :, 0]

    companion = cp.zeros(
        (signals.shape[0], m, m),
        dtype=coefficients.dtype,
    )
    companion[:, :-1, 1:] = cp.eye(m - 1, dtype=companion.dtype)
    companion[:, -1, :] = cp.flip(coefficients, axis=1)
    eigenvalues = cp.linalg.eigvals(companion)
    return _P1Result(
        singular_values=singular_values,
        coefficients=coefficients,
        eigenvalues=eigenvalues,
        selected_model_order=int(selected_model_order),
    )


def _linear_prediction_p2_impl(
    *,
    xp,
    time,
    trace,
    decay,
    angular_frequency,
) -> _P2Result:
    bins = xp.asarray(time, dtype=xp.float64)
    signal = xp.asarray(trace, dtype=xp.float64)
    decay = xp.asarray(decay, dtype=xp.float64)
    angular_frequency = xp.asarray(angular_frequency, dtype=xp.float64)
    if bins.ndim != 1 or signal.ndim != 1:
        raise ValueError("time and trace must be one-dimensional")
    if bins.shape != signal.shape:
        raise ValueError("time and trace must have the same shape")
    if decay.ndim != 1 or angular_frequency.ndim != 1:
        raise ValueError(
            "decay and angular_frequency must be one-dimensional"
        )
    if decay.shape != angular_frequency.shape:
        raise ValueError("decay and angular_frequency must have same shape")

    xbar, fit_time = _p2_design_matrix(
        xp,
        bins,
        decay,
        angular_frequency,
        dtype=signal.dtype,
    )
    coefficients = xp.linalg.lstsq(xbar, signal, rcond=None)[0]
    reconstruction = _p2_reconstruction(
        xp,
        fit_time=fit_time,
        decay=decay,
        angular_frequency=angular_frequency,
        coefficients=coefficients,
    )
    return _P2Result(
        coefficients=coefficients,
        reconstruction=reconstruction,
    )


def _linear_prediction_p2_artifacts_impl(
    *,
    xp,
    time,
    trace,
    decay,
    angular_frequency,
) -> _P2Artifacts:
    bins = xp.asarray(time, dtype=xp.float64)
    signal = xp.asarray(trace, dtype=xp.float64)
    decay = xp.asarray(decay, dtype=xp.float64)
    angular_frequency = xp.asarray(angular_frequency, dtype=xp.float64)
    if bins.ndim != 1 or signal.ndim != 1:
        raise ValueError("time and trace must be one-dimensional")
    if bins.shape != signal.shape:
        raise ValueError("time and trace must have the same shape")
    if decay.ndim != 1 or angular_frequency.ndim != 1:
        raise ValueError(
            "decay and angular_frequency must be one-dimensional"
        )
    if decay.shape != angular_frequency.shape:
        raise ValueError("decay and angular_frequency must have same shape")

    xbar, fit_time = _p2_design_matrix(
        xp,
        bins,
        decay,
        angular_frequency,
        dtype=signal.dtype,
    )
    coefficients = xp.linalg.lstsq(xbar, signal, rcond=None)[0]
    mode_counts = xp.asarray((int(decay.shape[0]),), dtype=xp.int64)
    artifacts = _p2_artifacts_from_coefficients_batched(
        xp,
        fit_time=fit_time,
        signals=signal[None, :],
        decay=decay[None, :],
        angular_frequency=angular_frequency[None, :],
        coefficients=coefficients[None, :],
        mode_counts=mode_counts,
    )
    return _P2Artifacts(
        coefficients=coefficients,
        amplitude=artifacts.amplitude[0],
        phase=artifacts.phase[0],
        time_components=artifacts.time_components[0],
        reconstruction=artifacts.reconstruction[0],
        frequency=artifacts.frequency[0],
        spectrum_components=artifacts.spectrum_components[0],
        spectrum_total=artifacts.spectrum_total[0],
        frequency_centers=artifacts.frequency_centers[0],
        chi2=artifacts.chi2[0],
    )


def _linear_prediction_p2_cupy_serial(
    time,
    traces,
    decay,
    angular_frequency,
):
    import cupy as cp

    results = [
        _linear_prediction_p2_impl(
            xp=cp,
            time=time,
            trace=trace,
            decay=decay,
            angular_frequency=angular_frequency,
        )
        for trace in traces
    ]
    return _P2Result(
        coefficients=cp.stack([item.coefficients for item in results]),
        reconstruction=cp.stack([item.reconstruction for item in results]),
    )


def _linear_prediction_p2_cupy_batched(
    time,
    traces,
    decay,
    angular_frequency,
):
    import cupy as cp

    bins = cp.asarray(time, dtype=cp.float64)
    signals = cp.asarray(traces, dtype=cp.float64)
    decay = cp.asarray(decay, dtype=cp.float64)
    angular_frequency = cp.asarray(angular_frequency, dtype=cp.float64)
    if bins.ndim != 1 or signals.ndim != 2:
        raise ValueError("time must be 1D and traces must be 2D")
    if signals.shape[1] != bins.shape[0]:
        raise ValueError("traces must have one row per time sample")
    if decay.ndim != 1 or angular_frequency.ndim != 1:
        raise ValueError(
            "decay and angular_frequency must be one-dimensional"
        )
    if decay.shape != angular_frequency.shape:
        raise ValueError("decay and angular_frequency must have same shape")

    xbar, fit_time = _p2_design_matrix(
        cp,
        bins,
        decay,
        angular_frequency,
        dtype=signals.dtype,
    )
    coefficients = cp.linalg.lstsq(xbar, signals.T, rcond=None)[0]
    reconstruction = _p2_reconstruction_batched(
        cp,
        fit_time=fit_time,
        decay=decay,
        angular_frequency=angular_frequency,
        coefficients=coefficients,
    )
    return _P2Result(
        coefficients=coefficients.T,
        reconstruction=reconstruction,
    )


def _p2_design_matrix(xp, bins, decay, angular_frequency, *, dtype):
    if bins.shape[0] < 2:
        raise ValueError("time must contain at least two samples")
    n = bins.shape[0] - 1
    delta_t = bins[1] - bins[0]
    fit_time = xp.linspace(0, n, n + 1) * delta_t
    nw = len(angular_frequency)
    exp_bt = xp.exp(-decay[:, None] * fit_time)
    cos_wt = xp.cos(angular_frequency[:, None] * fit_time)
    sin_wt = xp.sin(angular_frequency[:, None] * fit_time)

    xbar = xp.zeros((n + 1, nw * 2 + 1), dtype=dtype)
    xbar[:, :-1:2] = (exp_bt * cos_wt).T
    xbar[:, 1:-1:2] = (-exp_bt * sin_wt).T
    xbar[:, -1] = 1
    return xbar, fit_time


def _p2_reconstruction(
    xp,
    *,
    fit_time,
    decay,
    angular_frequency,
    coefficients,
):
    nw = len(angular_frequency)
    a0 = coefficients[: nw * 2 : 2]
    a1 = coefficients[1 : nw * 2 : 2]
    amplitude, phase = _amplitudes_and_phases_backend(xp, a0, a1)
    time_components = (
        amplitude * xp.exp(-decay * fit_time[:, None])
    ) * xp.cos(angular_frequency * fit_time[:, None] + phase)
    reconstruction = xp.sum(time_components, axis=1)
    return reconstruction + coefficients[2 * nw]


def _p2_reconstruction_batched(
    xp,
    *,
    fit_time,
    decay,
    angular_frequency,
    coefficients,
):
    nw = len(angular_frequency)
    a0 = coefficients[: nw * 2 : 2, :]
    a1 = coefficients[1 : nw * 2 : 2, :]
    amplitude, phase = _amplitudes_and_phases_backend(xp, a0, a1)
    time_components = (
        amplitude[None, :, :]
        * xp.exp(-decay[None, :, None] * fit_time[:, None, None])
    ) * xp.cos(
        angular_frequency[None, :, None] * fit_time[:, None, None]
        + phase[None, :, :]
    )
    reconstruction = xp.sum(time_components, axis=1)
    return (reconstruction + coefficients[2 * nw, :][None, :]).T


def _p2_artifacts_from_coefficients_batched(
    xp,
    *,
    fit_time,
    signals,
    decay,
    angular_frequency,
    coefficients,
    mode_counts,
) -> _P2Artifacts:
    row_count = int(signals.shape[0])
    sample_count = int(signals.shape[1])
    if sample_count < 3:
        raise ValueError("time must contain at least three samples")
    max_modes = int(decay.shape[1])
    active = (
        xp.arange(max_modes, dtype=xp.int64)[None, :] < mode_counts[:, None]
    )

    if max_modes:
        a0 = coefficients[:, : max_modes * 2 : 2]
        a1 = coefficients[:, 1 : max_modes * 2 : 2]
        amplitude, phase = _amplitudes_and_phases_backend(xp, a0, a1)
        amplitude = amplitude * active
        phase = phase * active
        time_components = (
            amplitude[:, None, :]
            * xp.exp(-decay[:, None, :] * fit_time[None, :, None])
        ) * xp.cos(
            angular_frequency[:, None, :] * fit_time[None, :, None]
            + phase[:, None, :]
        )
        time_components = time_components * active[:, None, :]
        reconstruction = xp.sum(time_components, axis=2)
        max_w = xp.max(
            xp.where(active, angular_frequency, 0),
            axis=1,
        )
        bi = decay / (2 * xp.pi)
        ww = angular_frequency / (2 * xp.pi)
        safe_bi = xp.where(active, bi, 1)
        safe_ww = xp.where(active, ww, 0)
    else:
        amplitude = xp.zeros((row_count, 0), dtype=signals.dtype)
        phase = xp.zeros((row_count, 0), dtype=signals.dtype)
        time_components = xp.zeros(
            (row_count, sample_count, 0),
            dtype=signals.dtype,
        )
        reconstruction = xp.zeros(
            (row_count, sample_count),
            dtype=signals.dtype,
        )
        max_w = xp.zeros((row_count,), dtype=signals.dtype)
        bi = xp.zeros((row_count, 0), dtype=signals.dtype)
        ww = xp.zeros((row_count, 0), dtype=signals.dtype)
        safe_bi = bi
        safe_ww = ww

    reconstruction = reconstruction + coefficients[:, -1:]
    frequency = (
        xp.linspace(0, 1, 1000, dtype=signals.dtype)[None, :]
        * (1.5 * xp.where(mode_counts > 0, max_w, 1e-5) / (2 * xp.pi))[
            :,
            None,
        ]
    )
    spectrum_components = xp.zeros(
        (row_count, frequency.shape[1], max_modes),
        dtype=signals.dtype,
    )
    if max_modes:
        spectrum_components = (amplitude * bi)[:, None, :] / (
            (safe_ww[:, None, :] - frequency[:, :, None]) ** 2
            + safe_bi[:, None, :] ** 2
        )
        spectrum_components = spectrum_components * active[:, None, :]
        max_indices = xp.argmax(spectrum_components, axis=1)
        frequency_centers = xp.take_along_axis(
            frequency[:, :, None],
            max_indices[:, None, :],
            axis=1,
        )[:, 0, :]
        frequency_centers = frequency_centers * active
    else:
        frequency_centers = xp.zeros((row_count, 0), dtype=signals.dtype)
    spectrum_total = xp.sum(spectrum_components, axis=2)
    residual = reconstruction[:, :-1] - signals[:, :-1]
    # Match _linear_prediction_impl and the reference wrapper: N is the last
    # sample index, so the production chi2 denominator is N - 1.
    chi2 = xp.sum(residual**2, axis=1) / (sample_count - 2)

    return _P2Artifacts(
        coefficients=coefficients,
        amplitude=amplitude,
        phase=phase,
        time_components=time_components,
        reconstruction=reconstruction,
        frequency=frequency,
        spectrum_components=spectrum_components,
        spectrum_total=spectrum_total,
        frequency_centers=frequency_centers,
        chi2=chi2,
    )


def _stack_p2_artifact_rows(
    xp,
    rows: list[_P2Artifacts],
    *,
    max_modes: int,
) -> _P2Artifacts:
    row_count = len(rows)
    if row_count == 0:
        raise ValueError("at least one artifact row is required")
    sample_count = int(rows[0].reconstruction.shape[0])
    frequency_count = int(rows[0].frequency.shape[0])
    coefficients = xp.zeros(
        (row_count, max_modes * 2 + 1),
        dtype=rows[0].reconstruction.dtype,
    )
    amplitude = xp.zeros((row_count, max_modes), dtype=coefficients.dtype)
    phase = xp.zeros_like(amplitude)
    time_components = xp.zeros(
        (row_count, sample_count, max_modes),
        dtype=coefficients.dtype,
    )
    reconstruction = xp.zeros(
        (row_count, sample_count),
        dtype=coefficients.dtype,
    )
    frequency = xp.zeros(
        (row_count, frequency_count),
        dtype=coefficients.dtype,
    )
    spectrum_components = xp.zeros(
        (row_count, frequency_count, max_modes),
        dtype=coefficients.dtype,
    )
    spectrum_total = xp.zeros(
        (row_count, frequency_count),
        dtype=coefficients.dtype,
    )
    frequency_centers = xp.zeros_like(amplitude)
    chi2 = xp.zeros((row_count,), dtype=coefficients.dtype)

    for row_index, row in enumerate(rows):
        mode_count = int(row.amplitude.shape[0])
        if mode_count:
            coefficients[row_index, : mode_count * 2] = row.coefficients[
                : mode_count * 2
            ]
            amplitude[row_index, :mode_count] = row.amplitude
            phase[row_index, :mode_count] = row.phase
            time_components[row_index, :, :mode_count] = row.time_components
            spectrum_components[row_index, :, :mode_count] = (
                row.spectrum_components
            )
            frequency_centers[row_index, :mode_count] = row.frequency_centers
        coefficients[row_index, -1] = row.coefficients[-1]
        reconstruction[row_index] = row.reconstruction
        frequency[row_index] = row.frequency
        spectrum_total[row_index] = row.spectrum_total
        chi2[row_index] = row.chi2

    return _P2Artifacts(
        coefficients=coefficients,
        amplitude=amplitude,
        phase=phase,
        time_components=time_components,
        reconstruction=reconstruction,
        frequency=frequency,
        spectrum_components=spectrum_components,
        spectrum_total=spectrum_total,
        frequency_centers=frequency_centers,
        chi2=chi2,
    )


def _p2_artifacts_to_numpy(xp, artifacts: _P2Artifacts) -> _P2Artifacts:
    return _P2Artifacts(
        coefficients=_to_numpy(xp, artifacts.coefficients),
        amplitude=_to_numpy(xp, artifacts.amplitude),
        phase=_to_numpy(xp, artifacts.phase),
        time_components=_to_numpy(xp, artifacts.time_components),
        reconstruction=_to_numpy(xp, artifacts.reconstruction),
        frequency=_to_numpy(xp, artifacts.frequency),
        spectrum_components=_to_numpy(xp, artifacts.spectrum_components),
        spectrum_total=_to_numpy(xp, artifacts.spectrum_total),
        frequency_centers=_to_numpy(xp, artifacts.frequency_centers),
        chi2=_to_numpy(xp, artifacts.chi2),
    )


def _max_abs_diff(left, right) -> float:
    diff = np.asarray(left) - np.asarray(right)
    if diff.size == 0:
        return 0.0
    return float(np.max(np.abs(diff)))


def _rms_diff(left, right) -> float:
    diff = np.asarray(left) - np.asarray(right)
    if diff.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.abs(diff) ** 2)))


def _mode_counts_tuple(xp, mode_counts):
    return tuple(int(item) for item in _to_numpy(xp, mode_counts))


def _legacy_tiles_max_abs_diff(xp, left_tiles, right_tiles, *, slot: int):
    max_diff = 0.0
    for left_tile, right_tile in zip(left_tiles, right_tiles, strict=True):
        for left_row, right_row in zip(left_tile, right_tile, strict=True):
            left = _to_numpy(xp, left_row[slot])
            right = _to_numpy(xp, right_row[slot])
            max_diff = max(max_diff, _max_abs_diff(left, right))
    return max_diff


def _assert_legacy_tiles_close(xp, left_tiles, right_tiles):
    for slot in (0, 2, 3, 9, 10, 11):
        diff = _legacy_tiles_max_abs_diff(
            xp, left_tiles, right_tiles, slot=slot
        )
        if not np.isfinite(diff) or diff > 1e-8:
            raise RuntimeError(
                "row-batched and multi-tile grouped legacy rows diverged "
                f"at slot {slot}: max_abs_diff={diff}"
            )


def _validate_savgol_benchmark_inputs(
    *,
    samples: int,
    window_length: int,
    polyorder: int,
    repeat: int,
):
    if repeat < 1:
        raise ValueError("repeat must be positive")
    if window_length <= 0 or window_length % 2 == 0:
        raise ValueError("window_length must be a positive odd integer")
    if polyorder < 0:
        raise ValueError("polyorder must be non-negative")
    if polyorder >= window_length:
        raise ValueError("polyorder must be smaller than window_length")
    if samples < window_length:
        raise ValueError("samples must be at least window_length")


def _validate_trace_batch(time, trace_rows):
    bins = np.asarray(time, dtype=np.float64)
    rows = np.asarray(trace_rows, dtype=np.float64)
    if bins.ndim != 1:
        raise ValueError("time must be one-dimensional")
    if rows.ndim != 2:
        raise ValueError("trace_rows must be two-dimensional")
    if rows.shape[0] < 1:
        raise ValueError("trace_rows must contain at least one row")
    if rows.shape[1] != bins.shape[0]:
        raise ValueError("trace_rows must have one column per time sample")
    return bins, rows


def _linear_prediction_mode_groups(mode_counts):
    counts: dict[int, int] = {}
    for mode_count in mode_counts:
        count = int(mode_count)
        counts[count] = counts.get(count, 0) + 1
    return tuple(
        LinearPredictionModeGroup(
            mode_count=mode_count,
            trace_count=trace_count,
            design_columns=int(mode_count * 2 + 1),
        )
        for mode_count, trace_count in sorted(counts.items())
    )


def _fixed_stages_numpy_serial(
    time,
    traces,
    decay,
    angular_frequency,
    *,
    window_length: int,
    polyorder: int,
    savgol_filter,
):
    filtered_rows = []
    p2_results = []
    for trace in traces:
        filtered = savgol_filter(trace, window_length, polyorder)
        filtered_rows.append(filtered)
        p2_results.append(
            _linear_prediction_p2_impl(
                xp=np,
                time=time,
                trace=filtered,
                decay=decay,
                angular_frequency=angular_frequency,
            )
        )
    return (
        np.stack(filtered_rows),
        _P2Result(
            coefficients=np.stack([item.coefficients for item in p2_results]),
            reconstruction=np.stack(
                [item.reconstruction for item in p2_results]
            ),
        ),
    )


def _fixed_stages_cupy_serial(
    time,
    traces,
    decay,
    angular_frequency,
    *,
    window_length: int,
    polyorder: int,
    savgol_filter,
):
    filtered = _savgol_cupy_serial(
        traces,
        window_length=window_length,
        polyorder=polyorder,
        savgol_filter=savgol_filter,
    )
    p2 = _linear_prediction_p2_cupy_serial(
        time,
        filtered,
        decay,
        angular_frequency,
    )
    return filtered, p2


def _fixed_stages_cupy_batched(
    time,
    traces,
    decay,
    angular_frequency,
    *,
    window_length: int,
    polyorder: int,
    savgol_filter,
):
    filtered = _savgol_cupy_batched(
        traces,
        window_length=window_length,
        polyorder=polyorder,
        savgol_filter=savgol_filter,
    )
    p2 = _linear_prediction_p2_cupy_batched(
        time,
        filtered,
        decay,
        angular_frequency,
    )
    return filtered, p2


def _variable_p2_numpy_serial(
    time,
    traces,
    decays: tuple[np.ndarray, ...],
    angular_frequencies: tuple[np.ndarray, ...],
):
    return np.stack(
        [
            _linear_prediction_p2_impl(
                xp=np,
                time=time,
                trace=trace,
                decay=decay,
                angular_frequency=angular_frequency,
            ).reconstruction
            for trace, decay, angular_frequency in zip(
                traces,
                decays,
                angular_frequencies,
                strict=True,
            )
        ]
    )


def _variable_stages_numpy_serial(
    time,
    traces,
    decays: tuple[np.ndarray, ...],
    angular_frequencies: tuple[np.ndarray, ...],
    *,
    window_length: int,
    polyorder: int,
    savgol_filter,
):
    filtered_rows = []
    reconstructions = []
    for trace, decay, angular_frequency in zip(
        traces,
        decays,
        angular_frequencies,
        strict=True,
    ):
        filtered = savgol_filter(trace, window_length, polyorder)
        filtered_rows.append(filtered)
        reconstructions.append(
            _linear_prediction_p2_impl(
                xp=np,
                time=time,
                trace=filtered,
                decay=decay,
                angular_frequency=angular_frequency,
            ).reconstruction
        )
    return np.stack(filtered_rows), np.stack(reconstructions)


def _variable_artifacts_numpy_serial(
    time,
    traces,
    decays: tuple[np.ndarray, ...],
    angular_frequencies: tuple[np.ndarray, ...],
    *,
    max_modes: int,
    window_length: int,
    polyorder: int,
    savgol_filter,
):
    filtered_rows = []
    artifact_rows = []
    for trace, decay, angular_frequency in zip(
        traces,
        decays,
        angular_frequencies,
        strict=True,
    ):
        filtered = savgol_filter(trace, window_length, polyorder)
        filtered_rows.append(filtered)
        artifact_rows.append(
            _linear_prediction_p2_artifacts_impl(
                xp=np,
                time=time,
                trace=filtered,
                decay=decay,
                angular_frequency=angular_frequency,
            )
        )
    return np.stack(filtered_rows), _stack_p2_artifact_rows(
        np,
        artifact_rows,
        max_modes=max_modes,
    )


def _pad_mode_arrays(
    decays: tuple[np.ndarray, ...],
    angular_frequencies: tuple[np.ndarray, ...],
    *,
    max_modes: int,
    xp=np,
):
    decay_padded = xp.zeros((len(decays), max_modes), dtype=xp.float64)
    angular_frequency_padded = xp.zeros_like(decay_padded)
    for row, (decay, angular_frequency) in enumerate(
        zip(decays, angular_frequencies, strict=True)
    ):
        if decay.shape != angular_frequency.shape:
            raise ValueError(
                "decay and angular_frequency must have same shape"
            )
        mode_count = int(decay.shape[0])
        decay_padded[row, :mode_count] = decay
        angular_frequency_padded[row, :mode_count] = angular_frequency
    return decay_padded, angular_frequency_padded


def _variable_p2_cupy_serial(
    time,
    traces,
    decay,
    angular_frequency,
    mode_counts: tuple[int, ...],
):
    import cupy as cp

    reconstructions = []
    for row, mode_count in enumerate(mode_counts):
        result = _linear_prediction_p2_impl(
            xp=cp,
            time=time,
            trace=traces[row],
            decay=decay[row, :mode_count],
            angular_frequency=angular_frequency[row, :mode_count],
        )
        reconstructions.append(result.reconstruction)
    return cp.stack(reconstructions)


def _variable_stages_cupy_serial(
    time,
    traces,
    decay,
    angular_frequency,
    mode_counts: tuple[int, ...],
    *,
    window_length: int,
    polyorder: int,
    savgol_filter,
):
    filtered = _savgol_cupy_serial(
        traces,
        window_length=window_length,
        polyorder=polyorder,
        savgol_filter=savgol_filter,
    )
    reconstruction = _variable_p2_cupy_serial(
        time,
        filtered,
        decay,
        angular_frequency,
        mode_counts,
    )
    return filtered, reconstruction


def _variable_artifacts_cupy_serial(
    time,
    traces,
    decay,
    angular_frequency,
    mode_counts: tuple[int, ...],
    *,
    max_modes: int,
    window_length: int,
    polyorder: int,
    savgol_filter,
):
    import cupy as cp

    filtered = _savgol_cupy_serial(
        traces,
        window_length=window_length,
        polyorder=polyorder,
        savgol_filter=savgol_filter,
    )
    artifact_rows = []
    for row, mode_count in enumerate(mode_counts):
        artifact_rows.append(
            _linear_prediction_p2_artifacts_impl(
                xp=cp,
                time=time,
                trace=filtered[row],
                decay=decay[row, :mode_count],
                angular_frequency=angular_frequency[row, :mode_count],
            )
        )
    return filtered, _stack_p2_artifact_rows(
        cp,
        artifact_rows,
        max_modes=max_modes,
    )


def _legacy_tiles_cupy_serial(
    time,
    tile_traces,
    tile_root_rows,
    tile_singular_value_rows,
    *,
    window_length: int,
    polyorder: int,
    savgol_filter,
):
    filtered_tiles = []
    legacy_tiles = []
    for traces, root_rows, singular_value_rows in zip(
        tile_traces,
        tile_root_rows,
        tile_singular_value_rows,
        strict=True,
    ):
        filtered, legacy_rows = _legacy_rows_cupy_serial(
            time,
            traces,
            root_rows,
            singular_value_rows,
            window_length=window_length,
            polyorder=polyorder,
            savgol_filter=savgol_filter,
        )
        filtered_tiles.append(filtered)
        legacy_tiles.append(legacy_rows)
    return tuple(filtered_tiles), tuple(legacy_tiles)


def _legacy_rows_cupy_serial(
    time,
    traces,
    root_rows,
    singular_value_rows,
    *,
    window_length: int,
    polyorder: int,
    savgol_filter,
):
    import cupy as cp

    root_rows = tuple(cp.asarray(row) for row in root_rows)
    singular_value_rows = tuple(
        cp.asarray(row, dtype=cp.float64) for row in singular_value_rows
    )
    modes = linear_prediction_mode_batch_from_roots_cupy(
        time,
        root_rows,
        singular_value_rows,
    )
    mode_counts = _mode_counts_tuple(cp, modes.mode_counts)
    filtered, artifacts = _variable_artifacts_cupy_serial(
        time,
        traces,
        modes.decay,
        modes.angular_frequency,
        mode_counts,
        max_modes=int(modes.decay.shape[1]),
        window_length=window_length,
        polyorder=polyorder,
        savgol_filter=savgol_filter,
    )
    legacy_rows = linear_prediction_legacy_rows_from_artifacts_cupy(
        time,
        artifacts,
        modes.decay,
        modes.angular_frequency,
        modes.mode_counts,
        singular_value_rows,
    )
    return filtered, legacy_rows


def _legacy_tiles_cupy_row_batched(
    time,
    tile_traces,
    tile_root_rows,
    tile_singular_value_rows,
    *,
    solver: str,
    window_length: int,
    polyorder: int,
):
    filtered_tiles = []
    legacy_tiles = []
    for traces, root_rows, singular_value_rows in zip(
        tile_traces,
        tile_root_rows,
        tile_singular_value_rows,
        strict=True,
    ):
        filtered, legacy_rows = linear_prediction_batched_legacy_rows_cupy(
            time,
            traces,
            root_rows,
            singular_value_rows,
            solver=solver,
            window_length=window_length,
            polyorder=polyorder,
        )
        filtered_tiles.append(filtered)
        legacy_tiles.append(legacy_rows)
    return tuple(filtered_tiles), tuple(legacy_tiles)


def _variable_stages_cupy_batched(
    time,
    traces,
    decay,
    angular_frequency,
    mode_counts,
    *,
    solver: str,
    window_length: int,
    polyorder: int,
    savgol_filter,
):
    filtered = _savgol_cupy_batched(
        traces,
        window_length=window_length,
        polyorder=polyorder,
        savgol_filter=savgol_filter,
    )
    reconstruction = _variable_p2_cupy_batched_solver(
        time,
        filtered,
        decay,
        angular_frequency,
        mode_counts,
        solver=solver,
    )
    return filtered, reconstruction


def _variable_artifacts_cupy_batched(
    time,
    traces,
    decay,
    angular_frequency,
    mode_counts,
    *,
    solver: str,
    window_length: int,
    polyorder: int,
    savgol_filter,
):
    filtered = _savgol_cupy_batched(
        traces,
        window_length=window_length,
        polyorder=polyorder,
        savgol_filter=savgol_filter,
    )
    artifacts = _variable_p2_artifacts_cupy_batched_solver(
        time,
        filtered,
        decay,
        angular_frequency,
        mode_counts,
        solver=solver,
    )
    return filtered, artifacts


def _variable_p2_cupy_batched_solver(
    time,
    traces,
    decay,
    angular_frequency,
    mode_counts,
    *,
    solver: str,
):
    if solver == "pinv":
        return _variable_p2_cupy_batched_pinv(
            time,
            traces,
            decay,
            angular_frequency,
            mode_counts,
        )
    if solver == "grouped-pinv":
        return _variable_p2_cupy_grouped_pinv(
            time,
            traces,
            decay,
            angular_frequency,
            mode_counts,
        )
    if solver == "grouped-normal":
        return _variable_p2_cupy_grouped_normal(
            time,
            traces,
            decay,
            angular_frequency,
            mode_counts,
        )
    if solver == "grouped-qr":
        return _variable_p2_cupy_grouped_qr(
            time,
            traces,
            decay,
            angular_frequency,
            mode_counts,
        )
    raise ValueError(
        "solver must be 'pinv', 'grouped-pinv', 'grouped-normal', "
        "or 'grouped-qr'"
    )


def _variable_p2_artifacts_cupy_batched_solver(
    time,
    traces,
    decay,
    angular_frequency,
    mode_counts,
    *,
    solver: str,
):
    if solver == "pinv":
        return _variable_p2_artifacts_cupy_batched_pinv(
            time,
            traces,
            decay,
            angular_frequency,
            mode_counts,
        )
    if solver == "grouped-pinv":
        return _variable_p2_artifacts_cupy_grouped_pinv(
            time,
            traces,
            decay,
            angular_frequency,
            mode_counts,
        )
    raise ValueError("solver must be 'pinv' or 'grouped-pinv'")


def _variable_p2_coefficients_cupy_batched_pinv(
    time,
    traces,
    decay,
    angular_frequency,
    mode_counts,
):
    import cupy as cp

    bins = cp.asarray(time, dtype=cp.float64)
    signals = cp.asarray(traces, dtype=cp.float64)
    decay = cp.asarray(decay, dtype=cp.float64)
    angular_frequency = cp.asarray(angular_frequency, dtype=cp.float64)
    mode_counts = cp.asarray(mode_counts, dtype=cp.int64)
    if bins.ndim != 1 or signals.ndim != 2:
        raise ValueError("time must be 1D and traces must be 2D")
    if signals.shape[1] != bins.shape[0]:
        raise ValueError("traces must have one row per time sample")
    if decay.shape != angular_frequency.shape:
        raise ValueError("decay and angular_frequency must have same shape")
    if decay.ndim != 2 or decay.shape[0] != signals.shape[0]:
        raise ValueError("mode arrays must have one row per trace")
    if mode_counts.shape != (signals.shape[0],):
        raise ValueError("mode_counts must have one item per trace")

    n = bins.shape[0] - 1
    delta_t = bins[1] - bins[0]
    fit_time = cp.linspace(0, n, n + 1) * delta_t
    max_modes = int(decay.shape[1])
    column_count = max_modes * 2 + 1
    xbar = cp.zeros(
        (signals.shape[0], bins.shape[0], column_count),
        dtype=signals.dtype,
    )
    if max_modes:
        active = (
            cp.arange(max_modes, dtype=cp.int64)[None, :]
            < mode_counts[:, None]
        )
        exp_bt = cp.exp(-decay[:, :, None] * fit_time[None, None, :])
        cos_wt = cp.cos(
            angular_frequency[:, :, None] * fit_time[None, None, :]
        )
        sin_wt = cp.sin(
            angular_frequency[:, :, None] * fit_time[None, None, :]
        )
        basis_cos = exp_bt * cos_wt * active[:, :, None]
        basis_sin = -exp_bt * sin_wt * active[:, :, None]
        xbar[:, :, : max_modes * 2 : 2] = cp.swapaxes(basis_cos, 1, 2)
        xbar[:, :, 1 : max_modes * 2 : 2] = cp.swapaxes(basis_sin, 1, 2)
    else:
        active = cp.zeros((signals.shape[0], 0), dtype=cp.bool_)
    xbar[:, :, -1] = 1

    coefficients = cp.matmul(
        cp.linalg.pinv(xbar),
        signals[:, :, None],
    )[:, :, 0]
    return fit_time, active, coefficients


def _variable_p2_cupy_batched_pinv(
    time,
    traces,
    decay,
    angular_frequency,
    mode_counts,
):
    import cupy as cp

    fit_time, active, coefficients = (
        _variable_p2_coefficients_cupy_batched_pinv(
            time,
            traces,
            decay,
            angular_frequency,
            mode_counts,
        )
    )
    decay = cp.asarray(decay, dtype=cp.float64)
    angular_frequency = cp.asarray(angular_frequency, dtype=cp.float64)
    return _variable_p2_reconstruction_batched(
        fit_time=fit_time,
        decay=decay,
        angular_frequency=angular_frequency,
        coefficients=coefficients,
        active=active,
    )


def _variable_p2_cupy_grouped_pinv(
    time,
    traces,
    decay,
    angular_frequency,
    mode_counts,
):
    import cupy as cp

    signals = cp.asarray(traces, dtype=cp.float64)
    mode_counts = cp.asarray(mode_counts, dtype=cp.int64)
    if signals.ndim != 2:
        raise ValueError("traces must be 2D")
    if mode_counts.shape != (signals.shape[0],):
        raise ValueError("mode_counts must have one item per trace")

    reconstruction = cp.empty_like(signals)
    for mode_count_item in cp.asnumpy(cp.unique(mode_counts)):
        mode_count = int(mode_count_item)
        row_indices = cp.where(mode_counts == mode_count)[0]
        group_decay = decay[row_indices, :mode_count]
        group_angular_frequency = angular_frequency[
            row_indices,
            :mode_count,
        ]
        group_reconstruction = _variable_p2_cupy_batched_pinv(
            time,
            signals[row_indices],
            group_decay,
            group_angular_frequency,
            cp.full(
                row_indices.shape,
                mode_count,
                dtype=cp.int64,
            ),
        )
        reconstruction[row_indices] = group_reconstruction
    return reconstruction


def _variable_p2_artifacts_cupy_batched_pinv(
    time,
    traces,
    decay,
    angular_frequency,
    mode_counts,
):
    import cupy as cp

    signals = cp.asarray(traces, dtype=cp.float64)
    decay = cp.asarray(decay, dtype=cp.float64)
    angular_frequency = cp.asarray(angular_frequency, dtype=cp.float64)
    mode_counts = cp.asarray(mode_counts, dtype=cp.int64)
    fit_time, _active, coefficients = (
        _variable_p2_coefficients_cupy_batched_pinv(
            time,
            signals,
            decay,
            angular_frequency,
            mode_counts,
        )
    )
    return _p2_artifacts_from_coefficients_batched(
        cp,
        fit_time=fit_time,
        signals=signals,
        decay=decay,
        angular_frequency=angular_frequency,
        coefficients=coefficients,
        mode_counts=mode_counts,
    )


def _variable_p2_artifacts_cupy_grouped_pinv(
    time,
    traces,
    decay,
    angular_frequency,
    mode_counts,
):
    import cupy as cp

    bins = cp.asarray(time, dtype=cp.float64)
    signals = cp.asarray(traces, dtype=cp.float64)
    decay = cp.asarray(decay, dtype=cp.float64)
    angular_frequency = cp.asarray(angular_frequency, dtype=cp.float64)
    mode_counts = cp.asarray(mode_counts, dtype=cp.int64)
    if bins.ndim != 1 or signals.ndim != 2:
        raise ValueError("time must be 1D and traces must be 2D")
    if signals.shape[1] != bins.shape[0]:
        raise ValueError("traces must have one row per time sample")
    if decay.shape != angular_frequency.shape:
        raise ValueError("decay and angular_frequency must have same shape")
    if decay.ndim != 2 or decay.shape[0] != signals.shape[0]:
        raise ValueError("mode arrays must have one row per trace")
    if mode_counts.shape != (signals.shape[0],):
        raise ValueError("mode_counts must have one item per trace")

    n = bins.shape[0] - 1
    delta_t = bins[1] - bins[0]
    fit_time = cp.linspace(0, n, n + 1) * delta_t
    max_modes = int(decay.shape[1])
    coefficients = cp.zeros(
        (signals.shape[0], max_modes * 2 + 1),
        dtype=signals.dtype,
    )
    for mode_count_item in cp.asnumpy(cp.unique(mode_counts)):
        mode_count = int(mode_count_item)
        row_indices = cp.where(mode_counts == mode_count)[0]
        group_decay = decay[row_indices, :mode_count]
        group_angular_frequency = angular_frequency[
            row_indices,
            :mode_count,
        ]
        _fit_time, _active, group_coefficients = (
            _variable_p2_coefficients_cupy_batched_pinv(
                bins,
                signals[row_indices],
                group_decay,
                group_angular_frequency,
                cp.full(
                    row_indices.shape,
                    mode_count,
                    dtype=cp.int64,
                ),
            )
        )
        if mode_count:
            coefficients[row_indices, : mode_count * 2] = group_coefficients[
                :,
                : mode_count * 2,
            ]
        coefficients[row_indices, -1] = group_coefficients[:, -1]

    return _p2_artifacts_from_coefficients_batched(
        cp,
        fit_time=fit_time,
        signals=signals,
        decay=decay,
        angular_frequency=angular_frequency,
        coefficients=coefficients,
        mode_counts=mode_counts,
    )


def _variable_p2_cupy_grouped_normal(
    time,
    traces,
    decay,
    angular_frequency,
    mode_counts,
):
    import cupy as cp

    bins = cp.asarray(time, dtype=cp.float64)
    signals = cp.asarray(traces, dtype=cp.float64)
    decay = cp.asarray(decay, dtype=cp.float64)
    angular_frequency = cp.asarray(angular_frequency, dtype=cp.float64)
    mode_counts = cp.asarray(mode_counts, dtype=cp.int64)
    if bins.ndim != 1 or signals.ndim != 2:
        raise ValueError("time must be 1D and traces must be 2D")
    if signals.shape[1] != bins.shape[0]:
        raise ValueError("traces must have one row per time sample")
    if decay.shape != angular_frequency.shape:
        raise ValueError("decay and angular_frequency must have same shape")
    if decay.ndim != 2 or decay.shape[0] != signals.shape[0]:
        raise ValueError("mode arrays must have one row per trace")
    if mode_counts.shape != (signals.shape[0],):
        raise ValueError("mode_counts must have one item per trace")

    n = bins.shape[0] - 1
    delta_t = bins[1] - bins[0]
    fit_time = cp.linspace(0, n, n + 1) * delta_t
    reconstruction = cp.empty_like(signals)
    for mode_count_item in cp.asnumpy(cp.unique(mode_counts)):
        mode_count = int(mode_count_item)
        row_indices = cp.where(mode_counts == mode_count)[0]
        group_decay = decay[row_indices, :mode_count]
        group_angular_frequency = angular_frequency[
            row_indices,
            :mode_count,
        ]
        xbar = _variable_p2_design_batched(
            fit_time,
            group_decay,
            group_angular_frequency,
            dtype=signals.dtype,
        )
        rhs = signals[row_indices, :, None]
        xbar_t = cp.swapaxes(xbar, 1, 2)
        normal_matrix = cp.matmul(xbar_t, xbar)
        normal_rhs = cp.matmul(xbar_t, rhs)
        coefficients = cp.linalg.solve(normal_matrix, normal_rhs)[:, :, 0]
        reconstruction[row_indices] = _variable_p2_reconstruction_batched(
            fit_time=fit_time,
            decay=group_decay,
            angular_frequency=group_angular_frequency,
            coefficients=coefficients,
            active=cp.ones(
                (row_indices.shape[0], mode_count),
                dtype=cp.bool_,
            ),
        )
    return reconstruction


def _variable_p2_cupy_grouped_qr(
    time,
    traces,
    decay,
    angular_frequency,
    mode_counts,
):
    import cupy as cp

    bins = cp.asarray(time, dtype=cp.float64)
    signals = cp.asarray(traces, dtype=cp.float64)
    decay = cp.asarray(decay, dtype=cp.float64)
    angular_frequency = cp.asarray(angular_frequency, dtype=cp.float64)
    mode_counts = cp.asarray(mode_counts, dtype=cp.int64)
    if bins.ndim != 1 or signals.ndim != 2:
        raise ValueError("time must be 1D and traces must be 2D")
    if signals.shape[1] != bins.shape[0]:
        raise ValueError("traces must have one row per time sample")
    if decay.shape != angular_frequency.shape:
        raise ValueError("decay and angular_frequency must have same shape")
    if decay.ndim != 2 or decay.shape[0] != signals.shape[0]:
        raise ValueError("mode arrays must have one row per trace")
    if mode_counts.shape != (signals.shape[0],):
        raise ValueError("mode_counts must have one item per trace")

    n = bins.shape[0] - 1
    delta_t = bins[1] - bins[0]
    fit_time = cp.linspace(0, n, n + 1) * delta_t
    reconstruction = cp.empty_like(signals)
    for mode_count_item in cp.asnumpy(cp.unique(mode_counts)):
        mode_count = int(mode_count_item)
        row_indices = cp.where(mode_counts == mode_count)[0]
        group_decay = decay[row_indices, :mode_count]
        group_angular_frequency = angular_frequency[
            row_indices,
            :mode_count,
        ]
        xbar = _variable_p2_design_batched(
            fit_time,
            group_decay,
            group_angular_frequency,
            dtype=signals.dtype,
        )
        q, r = cp.linalg.qr(xbar, mode="reduced")
        rhs = cp.matmul(
            cp.swapaxes(q, 1, 2),
            signals[row_indices, :, None],
        )
        coefficients = cp.linalg.solve(r, rhs)[:, :, 0]
        reconstruction[row_indices] = _variable_p2_reconstruction_batched(
            fit_time=fit_time,
            decay=group_decay,
            angular_frequency=group_angular_frequency,
            coefficients=coefficients,
            active=cp.ones(
                (row_indices.shape[0], mode_count),
                dtype=cp.bool_,
            ),
        )
    return reconstruction


def _variable_p2_design_batched(
    fit_time,
    decay,
    angular_frequency,
    *,
    dtype,
):
    import cupy as cp

    row_count = int(decay.shape[0])
    mode_count = int(decay.shape[1])
    column_count = mode_count * 2 + 1
    xbar = cp.empty((row_count, fit_time.shape[0], column_count), dtype=dtype)
    if mode_count:
        exp_bt = cp.exp(-decay[:, :, None] * fit_time[None, None, :])
        cos_wt = cp.cos(
            angular_frequency[:, :, None] * fit_time[None, None, :]
        )
        sin_wt = cp.sin(
            angular_frequency[:, :, None] * fit_time[None, None, :]
        )
        xbar[:, :, : mode_count * 2 : 2] = cp.swapaxes(exp_bt * cos_wt, 1, 2)
        xbar[:, :, 1 : mode_count * 2 : 2] = cp.swapaxes(
            -exp_bt * sin_wt,
            1,
            2,
        )
    xbar[:, :, -1] = 1
    return xbar


def _variable_p2_reconstruction_batched(
    *,
    fit_time,
    decay,
    angular_frequency,
    coefficients,
    active,
):
    import cupy as cp

    if active.shape[1] == 0:
        return cp.broadcast_to(
            coefficients[:, -1:],
            (coefficients.shape[0], fit_time.shape[0]),
        ).copy()

    a0 = coefficients[:, : active.shape[1] * 2 : 2]
    a1 = coefficients[:, 1 : active.shape[1] * 2 : 2]
    amplitude, phase = _amplitudes_and_phases_backend(cp, a0, a1)
    time_components = (
        amplitude[:, None, :]
        * cp.exp(-decay[:, None, :] * fit_time[None, :, None])
    ) * cp.cos(
        angular_frequency[:, None, :] * fit_time[None, :, None]
        + phase[:, None, :]
    )
    time_components = time_components * active[:, None, :]
    reconstruction = cp.sum(time_components, axis=2)
    return reconstruction + coefficients[:, -1:]


def _savgol_cupy_serial(
    traces,
    *,
    window_length: int,
    polyorder: int,
    savgol_filter,
):
    import cupy as cp

    return cp.stack(
        [
            savgol_filter(row, window_length, polyorder)
            for row in cp.asarray(traces)
        ]
    )


def _savgol_cupy_batched(
    traces,
    *,
    window_length: int,
    polyorder: int,
    savgol_filter,
):
    return savgol_filter(
        traces,
        window_length,
        polyorder,
        axis=1,
    )


def _validate_inputs(xp, time, trace, n_components):
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    bins = xp.asarray(time, dtype=xp.float64)
    signal = xp.asarray(trace, dtype=xp.float64)
    if bins.ndim != 1 or signal.ndim != 1:
        raise ValueError("time and trace must be one-dimensional")
    if bins.shape != signal.shape:
        raise ValueError("time and trace must have the same shape")
    if bins.shape[0] < 16:
        raise ValueError("time and trace must contain at least 16 samples")
    return bins, signal


def _component_count(xp, singular_values, *, requested: int, limit: int):
    components = min(int(requested), int(limit))
    if components <= 0:
        raise ValueError("n_components must be positive")
    max_s = xp.max(singular_values) if len(singular_values) else 0
    valid = xp.logical_and(
        xp.isfinite(singular_values),
        singular_values > max_s * 1e-12,
    )
    valid_components = _scalar_int(xp.count_nonzero(valid))
    components = min(components, valid_components)
    if components <= 0:
        raise ValueError("insufficient finite singular values")
    return components


def _prediction_roots_result(
    xp,
    coefficients,
    *,
    backend: str = "eigvals",
) -> _PredictionRootsResult:
    if backend not in {"eigvals", "roots"}:
        raise ValueError(f"unknown roots backend: {backend}")

    count = len(coefficients)
    array_module = _array_module_name(xp)
    if count == 0:
        return _PredictionRootsResult(
            roots=xp.asarray([], dtype=xp.complex128),
            stats=PredictionRootsStats(
                backend=backend,
                array_module=array_module,
                matrix_size=0,
                row_count=1,
                failures=0,
                elapsed_s=0.0,
            ),
        )

    sync = _sync_for_xp(xp)
    if sync is not None:
        sync()
    start = perf_counter()
    try:
        if backend == "eigvals":
            roots = xp.linalg.eigvals(_companion_matrix(xp, coefficients))
        else:
            roots = xp.roots(_prediction_polynomial(xp, coefficients))
        if sync is not None:
            sync()
    except Exception:
        if sync is not None:
            sync()
        raise
    elapsed_s = perf_counter() - start

    return _PredictionRootsResult(
        roots=roots,
        stats=PredictionRootsStats(
            backend=backend,
            array_module=array_module,
            matrix_size=count,
            row_count=1,
            failures=0,
            elapsed_s=elapsed_s,
        ),
    )


def _prediction_roots(xp, coefficients, *, backend: str = "eigvals"):
    return _prediction_roots_result(
        xp,
        coefficients,
        backend=backend,
    ).roots


def _prediction_polynomial(xp, coefficients):
    return xp.append(xp.asarray([1.0]), -1.0 * coefficients)


def _companion_matrix(xp, coefficients):
    count = len(coefficients)
    if count == 0:
        return xp.zeros((0, 0), dtype=xp.complex128)
    polynomial = _prediction_polynomial(xp, coefficients)
    normalized = polynomial[1:] / polynomial[0]
    return xp.append(
        xp.append(
            xp.zeros(count - 1, dtype=polynomial.dtype).reshape(count - 1, 1),
            xp.eye(count - 1, dtype=polynomial.dtype),
            axis=1,
        ),
        -1 * xp.flip(normalized).reshape(1, count),
        axis=0,
    )


def _batched_companion_matrix(xp, coefficients):
    coeffs = xp.asarray(coefficients)
    if coeffs.ndim != 2:
        raise ValueError("batched coefficients must be 2D")
    row_count, matrix_size = coeffs.shape
    companion = xp.zeros(
        (row_count, matrix_size, matrix_size),
        dtype=coeffs.dtype,
    )
    companion[:, :-1, 1:] = xp.eye(matrix_size - 1, dtype=companion.dtype)
    companion[:, -1, :] = xp.flip(coeffs, axis=1)
    return companion


def _roots_numpy_serial(coefficients, *, backend: str):
    return np.stack(
        [
            _sort_complex(
                _prediction_roots_result(
                    np,
                    coefficients_row,
                    backend=backend,
                ).roots
            )
            for coefficients_row in np.asarray(coefficients)
        ]
    )


def _benchmark_prediction_roots_backend(
    coefficients,
    *,
    backend_name: str,
    baseline_roots,
    repeat: int,
) -> PredictionRootsBackendBenchmark:
    matrix_size = int(coefficients.shape[1])
    row_count = int(coefficients.shape[0])
    try:
        best_s, roots = _time_best(
            lambda: _run_prediction_roots_backend(
                coefficients,
                backend_name=backend_name,
            ),
            repeat=repeat,
            sync=_sync_for_backend(backend_name),
        )
        root_diff = _max_abs_sorted_root_diff(baseline_roots, roots)
        return PredictionRootsBackendBenchmark(
            backend=backend_name,
            array_module=_backend_array_module_name(backend_name),
            batched=backend_name.endswith("-batched"),
            matrix_size=matrix_size,
            row_count=row_count,
            failures=0,
            best_s=float(best_s),
            max_abs_root_diff=float(root_diff),
            error=None,
        )
    except Exception as exc:
        return PredictionRootsBackendBenchmark(
            backend=backend_name,
            array_module=_backend_array_module_name(backend_name),
            batched=backend_name.endswith("-batched"),
            matrix_size=matrix_size,
            row_count=row_count,
            failures=row_count,
            best_s=None,
            max_abs_root_diff=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def _run_prediction_roots_backend(coefficients, *, backend_name: str):
    if backend_name == "numpy-eigvals":
        return _roots_numpy_serial(coefficients, backend="eigvals")
    if backend_name == "numpy-roots":
        return _roots_numpy_serial(coefficients, backend="roots")
    if backend_name == "cupy-eigvals-serial":
        return _roots_cupy_serial(coefficients, backend="eigvals")
    if backend_name == "cupy-roots-serial":
        return _roots_cupy_serial(coefficients, backend="roots")
    if backend_name == "cupy-eigvals-batched":
        return _roots_cupy_batched_eigvals(coefficients)
    raise ValueError(f"unknown roots benchmark backend: {backend_name}")


def _roots_cupy_serial(coefficients, *, backend: str):
    import cupy as cp

    gpu_coefficients = cp.asarray(coefficients)
    roots = [
        _prediction_roots_result(cp, row, backend=backend).roots
        for row in gpu_coefficients
    ]
    return np.stack([_sort_complex(cp.asnumpy(row)) for row in roots])


def _roots_cupy_batched_eigvals(coefficients):
    import cupy as cp

    gpu_coefficients = cp.asarray(coefficients)
    companion = _batched_companion_matrix(cp, gpu_coefficients)
    roots = cp.linalg.eigvals(companion)
    cp.cuda.Stream.null.synchronize()
    return np.stack([_sort_complex(row) for row in cp.asnumpy(roots)])


def _skipped_roots_backend(
    backend_name: str,
    coefficients,
    *,
    error: str,
) -> PredictionRootsBackendBenchmark:
    return PredictionRootsBackendBenchmark(
        backend=backend_name,
        array_module=_backend_array_module_name(backend_name),
        batched=backend_name.endswith("-batched"),
        matrix_size=int(coefficients.shape[1]),
        row_count=int(coefficients.shape[0]),
        failures=0,
        best_s=None,
        max_abs_root_diff=None,
        error=error,
    )


def _max_abs_sorted_root_diff(baseline_roots, roots):
    baseline = np.asarray(baseline_roots)
    candidate = np.asarray(roots)
    if baseline.shape != candidate.shape:
        raise ValueError(
            f"root shape mismatch: {baseline.shape} != {candidate.shape}"
        )
    if baseline.size == 0:
        return 0.0
    return float(np.max(np.abs(baseline - candidate)))


def _sync_for_backend(backend_name: str):
    if backend_name.startswith("cupy-"):
        import cupy as cp

        return cp.cuda.Stream.null.synchronize
    return None


def _array_module_name(xp):
    if xp is np:
        return "numpy"
    return getattr(xp, "__name__", type(xp).__name__)


def _backend_array_module_name(backend_name: str):
    if backend_name.startswith("cupy-"):
        return "cupy"
    if backend_name.startswith("numpy-"):
        return "numpy"
    return "unknown"


def _sync_for_xp(xp):
    if xp is np:
        return None
    stream = getattr(getattr(xp, "cuda", None), "Stream", None)
    if stream is None:
        return None
    return stream.null.synchronize


def _amplitudes_and_phases_backend(xp, a0, a1):
    if xp is np:
        return amplitudes_and_phases_numpy(a0, a1)
    return amplitudes_and_phases_cupy(a0, a1)


def _to_numpy(xp, value):
    if xp is np:
        return np.asarray(value)
    return xp.asnumpy(value)


def _scalar_int(value):
    if hasattr(value, "get"):
        value = value.get()
    return int(value)


def _with_elapsed(result: LinearPredictionResult, elapsed_s: float):
    return LinearPredictionResult(
        backend=result.backend,
        time=result.time,
        time_components=result.time_components,
        reconstruction=result.reconstruction,
        frequency=result.frequency,
        spectrum_components=result.spectrum_components,
        spectrum_total=result.spectrum_total,
        angular_frequency=result.angular_frequency,
        decay=result.decay,
        amplitude=result.amplitude,
        phase=result.phase,
        chi2=result.chi2,
        selected_model_order=result.selected_model_order,
        decaying_root_count=result.decaying_root_count,
        singular_values=result.singular_values,
        roots_stats=result.roots_stats,
        elapsed_s=elapsed_s,
    )


def _fit_stats_for_trace(
    time,
    trace,
    *,
    n_components: int,
    roots_backend: str,
) -> LinearPredictionFitStats:
    result = linear_prediction_numpy(
        time,
        trace,
        n_components,
        roots_backend=roots_backend,
    )
    return _fit_stats_from_result(
        result,
        trace,
        requested_components=n_components,
    )


def _fit_stats_from_result(
    result: LinearPredictionResult,
    trace,
    *,
    requested_components: int,
) -> LinearPredictionFitStats:
    residual = np.asarray(result.reconstruction) - np.asarray(trace)
    root_count = (
        0
        if result.roots_stats is None
        else int(result.roots_stats.matrix_size)
    )
    selected_model_order = int(result.selected_model_order)
    selected_root_count = int(len(result.angular_frequency))
    return LinearPredictionFitStats(
        requested_components=int(requested_components),
        selected_model_order=selected_model_order,
        matrix_size=root_count,
        root_count=root_count,
        selected_root_count=selected_root_count,
        decaying_root_count=int(result.decaying_root_count),
        filtered_root_count=max(root_count - selected_root_count, 0),
        chi2=float(result.chi2),
        rms_residual=float(np.sqrt(np.mean(np.abs(residual) ** 2))),
    )


def _singular_value_slice(values: np.ndarray, *, head: bool):
    if values.size == 0:
        return ()
    selected = values[:5] if head else values[-5:]
    return tuple(float(item) for item in selected)


def _singular_value_ratio(values: np.ndarray):
    finite = values[np.isfinite(values)]
    finite = finite[finite > 0]
    if finite.size < 2:
        return None
    return float(finite[-1] / finite[0])


def _time_best(function, *, repeat: int, sync=None):
    best_elapsed = None
    best_result = None
    for _idx in range(repeat):
        if sync is not None:
            sync()
        start = perf_counter()
        result = function()
        if sync is not None:
            sync()
        elapsed = perf_counter() - start
        if best_elapsed is None or elapsed < best_elapsed:
            best_elapsed = elapsed
            best_result = result
    return best_elapsed, best_result


def _sort_complex(values):
    return np.sort_complex(np.asarray(values))
