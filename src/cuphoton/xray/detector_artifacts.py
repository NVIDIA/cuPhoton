# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from cuphoton import __version__ as CUPHOTON_VERSION
from cuphoton.core.runtime import runtime_metadata

from ._types import FIT_DIAGNOSTICS_LEVELS, FitDiagnosticsLevel
from .detector_mask import (
    AxisRange,
    excluded_row_mask,
    format_y_ranges,
)
from .hdf5 import probe_hdf5_file
from .linear_prediction import (
    _linear_prediction_p1_impl,
    linear_prediction_cupy,
    linear_prediction_mode_batch_from_roots_cupy,
    linear_prediction_variable_artifacts_cupy_batched,
)
from .zero_offset import find_value_drop_position

if TYPE_CHECKING:
    from .iterative_fit import IterativeFitOptions


DETECTOR_ARRAYS = (
    "freq_all",
    "amp_all",
    "fft_all",
    "fft_freq_all",
    "amp_all_sum_filtered",
)
FIT_STATUS_FILE = "fit_status.npy"
FIT_DIAGNOSTICS_FILE = "fit_diagnostics.npz"
FIT_STATUS_UNPROCESSED = 0
FIT_STATUS_OK = 1
FIT_STATUS_SKIPPED = 2
FIT_STATUS_FAILED = 3
HDF5_READERS = ("h5py", "h5py-threaded", "hdf5-ts-funcwrap")
THREADED_HDF5_READERS = ("h5py-threaded", "hdf5-ts-funcwrap")
HDF5_TS_FUNCWRAP_REPOSITORY = "https://github.com/qkoziol/hdf5"
HDF5_TS_FUNCWRAP_BRANCH = "ts_funcwrap_1"
HDF5_TS_FUNCWRAP_REF = "4ac0f2fca8f68abf5ea25874832274a75b0f0967"
NORMALIZATION_MANIFEST_FILE = "normalization.json"
NORMALIZATION_CACHE_FILE = "normalization.npz"
DETECTOR_ARTIFACT_MANIFEST_VERSION = 2
DETECTOR_ARTIFACT_SUPPORTED_MANIFEST_VERSIONS = (1, 2, 3)
DETECTOR_NORMALIZATION_MANIFEST_VERSION = 1
FIT_DIAGNOSTICS_SCHEMA_VERSION = 1
ITERATIVE_FIT_DIAGNOSTICS_SCHEMA_VERSION = 2
_FULL_CONTENT_HASH_LIMIT = 16 * 1024 * 1024
_CONTENT_SAMPLE_BYTES = 64 * 1024

_FIT_DIAGNOSTIC_DTYPES = {
    "tile_x_start": np.int64,
    "tile_x_stop": np.int64,
    "tile_y_start": np.int64,
    "tile_y_stop": np.int64,
    "detector_y": np.int64,
    "fit_status": np.uint8,
    "trace_std": np.float64,
    "residual_std": np.float64,
    "relative_residual": np.float64,
    "chi2": np.float64,
    "selected_model_order": np.int64,
    "mode_count": np.int64,
    "p1_rank": np.int64,
    "p1_singular_value_ratio": np.float64,
    "p1_condition": np.float64,
    "p2_rank": np.int64,
    "p2_singular_value_ratio": np.float64,
    "p2_condition": np.float64,
    "max_amplitude": np.float64,
}
_FIT_DIAGNOSTIC_FULL_VALUE_FIELDS = (
    "trace",
    "reconstruction",
    "angular_frequency",
    "decay",
    "amplitude",
    "phase",
    "p1_singular_values",
    "p2_singular_values",
)
_FIT_DIAGNOSTIC_RAGGED_GROUPS = {
    "trace_offsets": ("trace",),
    "reconstruction_offsets": ("reconstruction",),
    "mode_offsets": (
        "angular_frequency",
        "decay",
        "amplitude",
        "phase",
    ),
    "p1_singular_value_offsets": ("p1_singular_values",),
    "p2_singular_value_offsets": ("p2_singular_values",),
}

_ITERATIVE_DIAGNOSTIC_DTYPES = {
    name: dtype
    for name, dtype in _FIT_DIAGNOSTIC_DTYPES.items()
    if not name.startswith(("p1_", "p2_")) and name != "selected_model_order"
}
_ITERATIVE_CONVERGENCE_DTYPES = {
    "converged": np.int8,
    "optimizer_status": np.dtype("U32"),
    "iterations": np.int64,
    "cost": np.float64,
    "gradient_norm": np.float64,
    "offset": np.float64,
}
_ITERATIVE_DIAGNOSTIC_DTYPES.update(_ITERATIVE_CONVERGENCE_DTYPES)


def validate_detector_fit_options(
    fit_method: str,
    iterative_options: IterativeFitOptions | None,
    p2_ridge_alpha: float,
) -> IterativeFitOptions | None:
    """Validate detector method controls before probing the GPU or inputs."""

    if fit_method not in {"linear-prediction", "iterative"}:
        raise ValueError("fit_method must be linear-prediction or iterative")
    if fit_method == "linear-prediction":
        if iterative_options is not None:
            raise ValueError(
                "iterative options require fit_method='iterative'"
            )
        return None
    if p2_ridge_alpha != 0.0:
        raise ValueError(
            "p2_ridge_alpha is only supported by linear prediction"
        )
    from .iterative_fit import IterativeFitOptions

    options = iterative_options
    if options is None:
        options = IterativeFitOptions()
    if not isinstance(options, IterativeFitOptions):
        raise ValueError("iterative_options must be an IterativeFitOptions")
    options.validate()
    return options


def _fit_diagnostic_fields(schema_version: int):
    if schema_version == FIT_DIAGNOSTICS_SCHEMA_VERSION:
        return (
            _FIT_DIAGNOSTIC_DTYPES,
            _FIT_DIAGNOSTIC_FULL_VALUE_FIELDS,
            _FIT_DIAGNOSTIC_RAGGED_GROUPS,
        )
    if schema_version != ITERATIVE_FIT_DIAGNOSTICS_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported fit diagnostics schema: {schema_version}"
        )
    return (
        _ITERATIVE_DIAGNOSTIC_DTYPES,
        tuple(
            name
            for name in _FIT_DIAGNOSTIC_FULL_VALUE_FIELDS
            if not name.startswith(("p1_", "p2_"))
        ),
        {
            name: fields
            for name, fields in _FIT_DIAGNOSTIC_RAGGED_GROUPS.items()
            if not name.startswith(("p1_", "p2_"))
        },
    )


@dataclass(frozen=True)
class DetectorArtifactResult:
    """Completed detector-wide artifact build.

    ``shape`` is ``(roi_y, roi_x, spectral_bins)`` for emitted detector
    arrays, ``roi_lower`` and ``roi_dim`` use ``(x, y)`` ordering in detector
    pixels, and ``elapsed_s`` is wall time in seconds. ``batched_tiles``
    counts the processed tiles whose fitted rows went through the batched
    artifact path rather than the row-by-row fallback. Output paths own the
    persisted arrays; this result contains metadata only.
    """

    output_dir: Path
    manifest_path: Path
    shape: tuple[int, int, int]
    roi_lower: tuple[int, int]
    roi_dim: tuple[int, int]
    normalization_shift: float
    zero_offset_index: int
    zero_offset_status: str
    processed_tiles: int
    batched_tiles: int
    raw_fits: int
    failures: int
    skipped_fits: int
    filtered_fits: int
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["manifest_path"] = str(self.manifest_path)
        return payload


@dataclass(frozen=True)
class DetectorNormalizationResult:
    """Reusable detector-normalization cache and its provenance.

    ``detector_shape`` uses ``(height, width)`` pixels. ``sample_count`` is
    the number of retained frames, and ``elapsed_s`` is wall time in seconds.
    """

    output_dir: Path
    manifest_path: Path
    cache_path: Path
    detector_shape: tuple[int, int]
    sample_count: int
    normalization_shift: float
    zero_offset_index: int
    zero_offset_status: str
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["manifest_path"] = str(self.manifest_path)
        payload["cache_path"] = str(self.cache_path)
        return payload


@dataclass(frozen=True)
class DetectorArtifactMergeResult:
    """Merged detector-artifact shards.

    ``shape`` is ``(roi_y, roi_x, spectral_bins)``, while ``roi_lower`` and
    ``roi_dim`` use ``(x, y)`` detector-pixel ordering. ``elapsed_s`` reports
    merge wall time in seconds.
    """

    output_dir: Path
    manifest_path: Path
    shard_count: int
    shape: tuple[int, int, int]
    roi_lower: tuple[int, int]
    roi_dim: tuple[int, int]
    elapsed_s: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_dir"] = str(self.output_dir)
        payload["manifest_path"] = str(self.manifest_path)
        return payload


def build_detector_artifacts_cupy(
    *,
    h5dir: Path | str,
    fon: str,
    foff: str,
    output_dir: Path | str,
    roi_lower: tuple[int, int] = (0, 0),
    roi_dim: tuple[int, int] | None = None,
    tile_shape: tuple[int, int] = (16, 16),
    exclude_y: tuple[AxisRange, ...] = (),
    drop_leading: int = 1,
    chunk_frames: int = 16,
    zero_offset: float = 0.0,
    zero_offset_index: int | None = None,
    fit_trailing_drop: int = 1,
    integrate_pixels: int = 3,
    components: int = 30,
    fit_method: str = "linear-prediction",
    iterative_options: IterativeFitOptions | None = None,
    p2_ridge_alpha: float = 0.0,
    roots_backend: str = "eigvals",
    batch_rows: bool = True,
    savgol_window: int = 5,
    savgol_polyorder: int = 3,
    amp_threshold: float = 1.6,
    max_fit_failures: int = 0,
    hdf5_reader: str = "h5py",
    hdf5_reader_workers: int = 2,
    max_tiles: int | None = None,
    normalization_cache: Path | str | None = None,
    fit_diagnostics: FitDiagnosticsLevel = "none",
    shard_index: int | None = None,
    shard_count: int | None = None,
    global_roi_lower: tuple[int, int] | None = None,
    global_roi_dim: tuple[int, int] | None = None,
) -> DetectorArtifactResult:
    """Write detector-wide LPF arrays using HDF5 input and CuPy fitting.

    The emitted arrays match the dumped reference-analysis contract:
    ``freq_all.npy``, ``amp_all.npy``, ``fft_all.npy``, ``fft_freq_all.npy``,
    and ``amp_all_sum_filtered.npy``. Each fitted detector row is broadcast
    across the x-columns of its tile, preserving the current vertical-strip
    science path.

    Set ``batch_rows=False`` to force the serial row loop for A/B checks
    and diagnostics. The default batches eligible rows and retains the
    existing row-local fallback. The chosen mode is recorded in the manifest
    and distinguished by its configuration and resume identities.
    """

    _require_positive("tile width", tile_shape[0])
    _require_positive("tile height", tile_shape[1])
    _require_nonnegative("drop_leading", drop_leading)
    _require_positive("chunk_frames", chunk_frames)
    _require_nonnegative("fit_trailing_drop", fit_trailing_drop)
    _require_nonnegative("integrate_pixels", integrate_pixels)
    _require_positive("components", components)
    _require_finite_nonnegative("p2_ridge_alpha", p2_ridge_alpha)
    iterative_options = validate_detector_fit_options(
        fit_method, iterative_options, p2_ridge_alpha
    )
    _require_positive("savgol_window", savgol_window)
    _require_nonnegative("savgol_polyorder", savgol_polyorder)
    _require_nonnegative("max_fit_failures", max_fit_failures)
    _require_positive("hdf5_reader_workers", hdf5_reader_workers)
    _validate_hdf5_reader(hdf5_reader, hdf5_reader_workers)
    _validate_fit_diagnostics(fit_diagnostics)
    if savgol_window % 2 != 1:
        raise ValueError("savgol_window must be odd")
    if savgol_polyorder >= savgol_window:
        raise ValueError(
            "savgol_polyorder must be smaller than savgol_window"
        )
    if max_tiles is not None:
        _require_positive("max_tiles", max_tiles)
    if (shard_index is None) != (shard_count is None):
        raise ValueError(
            "shard_index and shard_count must be provided together"
        )
    if shard_index is not None:
        _require_nonnegative("shard_index", shard_index)
        _require_positive("shard_count", shard_count)
        if shard_index >= shard_count:
            raise ValueError("shard_index must be smaller than shard_count")
    if (global_roi_lower is None) != (global_roi_dim is None):
        raise ValueError(
            "global_roi_lower and global_roi_dim must be provided together"
        )

    try:
        import cupy as cp
        from cupyx.scipy.signal import savgol_filter as savgol_filter_cupy
    except ImportError as exc:
        raise RuntimeError(
            "CuPy is required for detector artifacts; run "
            "'uv sync --extra gpu' for development or install "
            "'cuphoton[gpu]'"
        ) from exc

    _ensure_cuda_device(cp)

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for detector artifacts") from exc

    root = Path(h5dir)
    on_path = _resolve_path(root, fon)
    off_path = _resolve_path(root, foff)
    input_identity = detector_artifact_input_identity(on_path, off_path)
    normalization_identity = detector_normalization_identity(
        normalization_cache
    )
    on_probe = probe_hdf5_file(on_path)
    off_probe = probe_hdf5_file(off_path)
    if on_probe.load_plan is None or off_probe.load_plan is None:
        raise ValueError("unsupported HDF5 schema or IPM layout")
    if on_probe.load_plan.schema != off_probe.load_plan.schema:
        raise ValueError("on/off HDF5 schemas do not match")
    plan = on_probe.load_plan

    start_time = perf_counter()
    target_output = Path(output_dir)
    output = target_output
    staging_dir: tempfile.TemporaryDirectory[str] | None = None

    with h5py.File(on_path, "r") as on_h5, h5py.File(off_path, "r") as off_h5:
        on_image = on_h5[plan.image_dataset]
        off_image = off_h5[plan.image_dataset]
        if len(on_image.shape) != 3 or len(off_image.shape) != 3:
            raise ValueError(
                "detector artifact input requires 3D image cubes"
            )
        if tuple(on_image.shape) != tuple(off_image.shape):
            raise ValueError("on/off image shapes do not match")
        frame_count, detector_height, detector_width = (
            int(on_image.shape[0]),
            int(on_image.shape[1]),
            int(on_image.shape[2]),
        )
        if drop_leading >= frame_count:
            raise ValueError("drop_leading removes all frames")

        sample_count = frame_count - drop_leading
        if savgol_window > sample_count:
            raise ValueError(
                "savgol_window must be less than or equal to usable samples"
            )
        delay = np.asarray(
            on_h5[plan.delay_dataset][drop_leading:],
            dtype=np.float64,
        )
        off_delay = np.asarray(
            off_h5[plan.delay_dataset][drop_leading:],
            dtype=np.float64,
        )
        if delay.shape != (sample_count,) or off_delay.shape != delay.shape:
            raise ValueError("delay length does not match image frames")
        if not np.allclose(delay, off_delay):
            raise ValueError("on/off delay axes do not match")

        on_norm = np.asarray(
            on_h5[plan.ipm_pair.normalization][drop_leading:],
            dtype=np.float64,
        )
        off_norm = np.asarray(
            off_h5[plan.ipm_pair.normalization][drop_leading:],
            dtype=np.float64,
        )
        if on_norm.shape != (sample_count,) or off_norm.shape != (
            sample_count,
        ):
            raise ValueError("normalization length does not match frames")

        roi_x, roi_y, roi_width, roi_height = _resolve_roi(
            roi_lower=roi_lower,
            roi_dim=roi_dim,
            detector_width=detector_width,
            detector_height=detector_height,
        )
        if fit_trailing_drop >= sample_count:
            raise ValueError("fit_trailing_drop removes all samples")
        padded_length = sample_count // 2
        if padded_length < 1:
            raise ValueError("detector artifact run needs at least 2 samples")

        with _open_hdf5_block_reader(
            mode=hdf5_reader,
            on_path=on_path,
            off_path=off_path,
            image_dataset=plan.image_dataset,
            on_dataset=on_image,
            off_dataset=off_image,
            workers=hdf5_reader_workers,
        ) as block_reader:
            normalization_cache_payload = None
            if normalization_cache is None:
                full_trace = _full_detector_trace(
                    block_reader,
                    on_norm,
                    off_norm,
                    detector_height=detector_height,
                    detector_width=detector_width,
                    frame_count=frame_count,
                    drop_leading=drop_leading,
                    chunk_frames=chunk_frames,
                )
                normalization_shift = full_trace["normalization_shift"]
                if zero_offset_index is None:
                    zero_result = find_value_drop_position(
                        delay,
                        full_trace["ratio_minus_one"],
                        zero_offset=zero_offset,
                    )
                    fit_start = int(zero_result.selected_index)
                    if fit_start < 0:
                        raise ValueError(
                            "zero offset could not be found "
                            f"({zero_result.status}); pass "
                            "--zero-offset-index for a diagnostic or "
                            "cropped run"
                        )
                    zero_result_payload = asdict(zero_result)
                else:
                    fit_start = int(zero_offset_index)
                    zero_result_payload = None
            else:
                normalization_payload = _load_detector_normalization_cache(
                    normalization_cache
                )
                _validate_detector_normalization_cache(
                    normalization_payload,
                    input_identity=input_identity,
                    schema=plan.schema,
                    image_dataset=plan.image_dataset,
                    delay_dataset=plan.delay_dataset,
                    normalization_dataset=plan.ipm_pair.normalization,
                    detector_height=detector_height,
                    detector_width=detector_width,
                    frame_count=frame_count,
                    sample_count=sample_count,
                    drop_leading=drop_leading,
                    zero_offset=zero_offset,
                    zero_offset_index=zero_offset_index,
                    delay=delay,
                    on_norm=on_norm,
                    off_norm=off_norm,
                )
                cache_manifest = normalization_payload["manifest"]
                normalization_shift = float(
                    cache_manifest["normalization_shift"]
                )
                fit_start = int(cache_manifest["zero_offset_index"])
                zero_result_payload = cache_manifest.get("zero_offset_result")
                normalization_cache_payload = {
                    "identity": normalization_identity,
                    "config_hash": cache_manifest.get("config_hash"),
                }
            if fit_start < 0 or fit_start + fit_trailing_drop >= sample_count:
                raise ValueError(
                    "zero_offset_index leaves no usable fit samples"
                )
            if sample_count - fit_start - fit_trailing_drop < 16:
                raise ValueError("fit sample window is too short")

            target_output.parent.mkdir(parents=True, exist_ok=True)
            staging_dir = tempfile.TemporaryDirectory(
                prefix=f".{target_output.name}.tmp-",
                dir=target_output.parent,
            )
            output = Path(staging_dir.name)
            shape = (roi_height, roi_width, padded_length)
            freq_all = _open_output_array(output / "freq_all.npy", shape)
            amp_all = _open_output_array(output / "amp_all.npy", shape)
            fft_all = _open_output_array(output / "fft_all.npy", shape)
            fft_freq_all = _open_output_array(
                output / "fft_freq_all.npy",
                shape,
            )
            fit_status = _open_output_array(
                output / FIT_STATUS_FILE,
                (roi_height, roi_width),
                dtype=np.uint8,
            )
            _zero_output_arrays(
                freq_all,
                amp_all,
                fft_all,
                fft_freq_all,
                fit_status,
            )

            stats = _RunStats()
            diagnostics_writer = _FitDiagnosticsWriter(
                output,
                fit_diagnostics,
                schema_version=(
                    ITERATIVE_FIT_DIAGNOSTICS_SCHEMA_VERSION
                    if fit_method == "iterative"
                    else FIT_DIAGNOSTICS_SCHEMA_VERSION
                ),
            )
            fit_error_types = _fit_error_types(cp)
            for tile_index, (x0, x1, y0, y1) in enumerate(
                _iter_tiles(
                    roi_x=roi_x,
                    roi_y=roi_y,
                    roi_width=roi_width,
                    roi_height=roi_height,
                    tile_width=tile_shape[0],
                    tile_height=tile_shape[1],
                )
            ):
                if max_tiles is not None and tile_index >= max_tiles:
                    break
                read_y0, read_y1 = _row_halo_bounds(
                    y0,
                    y1,
                    detector_height=detector_height,
                    pixels=integrate_pixels,
                )
                sig_diff = _load_tile_signal(
                    block_reader,
                    on_norm,
                    off_norm,
                    shift=normalization_shift,
                    drop_leading=drop_leading,
                    x0=x0,
                    x1=x1,
                    y0=read_y0,
                    y1=read_y1,
                )
                skip_rows = excluded_row_mask(y0, y1, exclude_y)
                _zero_excluded_signal_rows(
                    sig_diff,
                    y0=read_y0,
                    y1=read_y1,
                    exclude_y=exclude_y,
                )
                # Integrate the padded row block after masking excluded source
                # rows, then skip excluded row predictions and outputs after
                # cropping back to the tile.
                padded_traces_gpu = _integrate_cupy(
                    cp,
                    cp.asarray(sig_diff, dtype=cp.float64),
                    integrate_pixels,
                )
                crop_start = y0 - read_y0
                crop_stop = crop_start + (y1 - y0)
                traces_gpu = padded_traces_gpu[:, crop_start:crop_stop]
                filtered_gpu = savgol_filter_cupy(
                    traces_gpu,
                    savgol_window,
                    savgol_polyorder,
                    axis=0,
                )
                fit_time_gpu = cp.asarray(
                    delay[fit_start : sample_count - fit_trailing_drop],
                    dtype=cp.float64,
                )
                active_local_rows = tuple(
                    row for row, skip in enumerate(skip_rows) if not skip
                )
                batched_rows: dict[int, dict[str, Any] | None] = {}
                if (
                    batch_rows
                    and fit_method == "linear-prediction"
                    and fit_diagnostics == "none"
                    and p2_ridge_alpha == 0.0
                    and roots_backend == "eigvals"
                    and len(active_local_rows) > 1
                ):
                    active_indices_gpu = cp.asarray(
                        active_local_rows,
                        dtype=cp.int64,
                    )
                    active_traces_gpu = cp.ascontiguousarray(
                        filtered_gpu[
                            fit_start : sample_count - fit_trailing_drop
                        ].T[active_indices_gpu]
                    )
                    try:
                        rows = _fit_detector_rows_batched(
                            cp=cp,
                            time_gpu=fit_time_gpu,
                            traces_gpu=active_traces_gpu,
                            components=components,
                            padded_length=padded_length,
                        )
                    except fit_error_types:
                        # Preserve row-local failure accounting when a tile
                        # cannot use the optimized batch path.
                        pass
                    else:
                        batched_rows = dict(
                            zip(active_local_rows, rows, strict=True)
                        )
                        if any(
                            row is not None for row in batched_rows.values()
                        ):
                            stats.batched_tiles += 1
                for local_row, skip in enumerate(skip_rows):
                    output_y = y0 + local_row - roi_y
                    output_x = slice(x0 - roi_x, x1 - roi_x)
                    if skip:
                        fit_status[output_y, output_x] = FIT_STATUS_SKIPPED
                        stats.skipped_fits += 1
                        _append_non_ok_fit_diagnostic(
                            diagnostics_writer,
                            level=fit_diagnostics,
                            fit_status=FIT_STATUS_SKIPPED,
                            x0=x0,
                            x1=x1,
                            y0=y0,
                            y1=y1,
                            detector_y=y0 + local_row,
                        )
                        continue
                    row = batched_rows.get(local_row)
                    if row is None:
                        trace_gpu = filtered_gpu[
                            fit_start : sample_count - fit_trailing_drop,
                            local_row,
                        ]
                        try:
                            row = _fit_detector_row(
                                cp=cp,
                                time_gpu=fit_time_gpu,
                                trace_gpu=trace_gpu,
                                components=components,
                                fit_method=fit_method,
                                iterative_options=iterative_options,
                                p2_ridge_alpha=p2_ridge_alpha,
                                roots_backend=roots_backend,
                                padded_length=padded_length,
                                fit_diagnostics=fit_diagnostics,
                            )
                        except fit_error_types as exc:
                            fit_status[output_y, output_x] = FIT_STATUS_FAILED
                            stats.failures += 1
                            _append_non_ok_fit_diagnostic(
                                diagnostics_writer,
                                level=fit_diagnostics,
                                fit_status=FIT_STATUS_FAILED,
                                iterative_result=getattr(
                                    exc, "iterative_result", None
                                ),
                                x0=x0,
                                x1=x1,
                                y0=y0,
                                y1=y1,
                                detector_y=y0 + local_row,
                            )
                            if stats.failures > max_fit_failures:
                                raise RuntimeError(
                                    "detector artifact generation had "
                                    f"{stats.failures} fit failures; allowed "
                                    f"{max_fit_failures}"
                                    + (
                                        f"; last iterative fit: {exc}"
                                        if fit_method == "iterative"
                                        else ""
                                    )
                                ) from exc
                            continue
                    _write_row_outputs(
                        freq_all,
                        amp_all,
                        fft_all,
                        fft_freq_all,
                        output_y=output_y,
                        output_x=output_x,
                        row=row,
                    )
                    fit_status[output_y, output_x] = FIT_STATUS_OK
                    stats.raw_fits += 1
                    if fit_diagnostics != "none":
                        diagnostics_writer.append(
                            {
                                "tile_x_start": x0,
                                "tile_x_stop": x1,
                                "tile_y_start": y0,
                                "tile_y_stop": y1,
                                "detector_y": y0 + local_row,
                                **row["diagnostics"],
                            }
                        )
                    if np.any(
                        (row["amp"] > amp_threshold) & (row["amp"] < 1e6)
                    ):
                        stats.filtered_fits += 1
                stats.processed_tiles += 1

            for array in (
                freq_all,
                amp_all,
                fft_all,
                fft_freq_all,
                fit_status,
            ):
                array.flush()

            if stats.raw_fits < 1:
                raise RuntimeError(
                    "detector artifact generation produced no successful "
                    "row fits"
                )
            if stats.failures > max_fit_failures:
                raise RuntimeError(
                    "detector artifact generation had "
                    f"{stats.failures} fit failures; allowed "
                    f"{max_fit_failures}"
                )

            amp_sum = _write_amp_sum_filtered(
                output / "amp_all_sum_filtered.npy",
                amp_all,
                amp_threshold=amp_threshold,
            )
            amp_sum.flush()
            fit_diagnostics_metadata = diagnostics_writer.finalize(
                source_identity_sha256=_stable_json_hash(input_identity)
            )

    elapsed = perf_counter() - start_time
    runtime = runtime_metadata(backend="cupy", dtype="float64")
    manifest = {
        "kind": "xray-detector-artifacts",
        "manifest_schema_version": (
            3
            if fit_method == "iterative"
            else DETECTOR_ARTIFACT_MANIFEST_VERSION
        ),
        "package_version": runtime["package_version"],
        "backend": "cupy",
        "device": runtime["device"],
        "dtype": runtime["dtype"],
        "runtime": runtime,
        "input_identity": input_identity,
        "normalization_identity": normalization_identity,
        "schema": plan.schema,
        "image_dataset": plan.image_dataset,
        "delay_dataset": plan.delay_dataset,
        "normalization_dataset": plan.ipm_pair.normalization,
        "drop_leading": int(drop_leading),
        "chunk_frames": int(chunk_frames),
        "fit_trailing_drop": int(fit_trailing_drop),
        "zero_offset": float(zero_offset),
        "requested_zero_offset_index": (
            None if zero_offset_index is None else int(zero_offset_index)
        ),
        "zero_offset_index": int(fit_start),
        "zero_offset_result": zero_result_payload,
        "normalization_shift": float(normalization_shift),
        "normalization_cache": normalization_cache_payload,
        "detector_shape": [int(detector_height), int(detector_width)],
        "roi_lower": [int(roi_x), int(roi_y)],
        "roi_dim": [int(roi_width), int(roi_height)],
        "output_shape": [int(value) for value in shape],
        "tile_shape": [int(tile_shape[0]), int(tile_shape[1])],
        "exclude_y": format_y_ranges(exclude_y),
        "integrate_pixels": int(integrate_pixels),
        "components": int(components),
        "p2_ridge_alpha": float(p2_ridge_alpha),
        "roots_backend": roots_backend,
        "batch_rows": bool(batch_rows),
        "savgol_window": int(savgol_window),
        "savgol_polyorder": int(savgol_polyorder),
        "amp_threshold": float(amp_threshold),
        "max_fit_failures": int(max_fit_failures),
        "hdf5_reader": hdf5_reader,
        "hdf5_reader_runtime": _hdf5_reader_runtime(
            hdf5_reader=hdf5_reader,
            h5py=h5py,
        ),
        "hdf5_reader_workers": int(hdf5_reader_workers),
        "max_tiles": None if max_tiles is None else int(max_tiles),
        "processed_tiles": int(stats.processed_tiles),
        "batched_tiles": int(stats.batched_tiles),
        "raw_fits": int(stats.raw_fits),
        "failures": int(stats.failures),
        "skipped_fits": int(stats.skipped_fits),
        "filtered_fits": int(stats.filtered_fits),
        "elapsed_s": float(elapsed),
        "fit_diagnostics": fit_diagnostics_metadata,
        "arrays": [f"{name}.npy" for name in DETECTOR_ARRAYS]
        + [FIT_STATUS_FILE],
        "fit_status_codes": {
            "unprocessed": FIT_STATUS_UNPROCESSED,
            "ok": FIT_STATUS_OK,
            "skipped": FIT_STATUS_SKIPPED,
            "failed": FIT_STATUS_FAILED,
        },
    }
    if fit_method == "iterative":
        manifest["fit_method"] = fit_method
        manifest["iterative_options"] = iterative_options.to_dict()
        manifest["frequency_semantics"] = (
            "fitted modal frequencies in cycles per delay unit"
        )
    if shard_index is not None:
        manifest["shard"] = {
            "index": int(shard_index),
            "count": int(shard_count),
            "global_roi_lower": (
                [int(global_roi_lower[0]), int(global_roi_lower[1])]
                if global_roi_lower is not None
                else [int(roi_x), int(roi_y)]
            ),
            "global_roi_dim": (
                [int(global_roi_dim[0]), int(global_roi_dim[1])]
                if global_roi_dim is not None
                else [int(roi_width), int(roi_height)]
            ),
            "x_range": [int(roi_x), int(roi_x + roi_width)],
        }
    manifest["resume_identity"] = detector_artifact_resume_identity(manifest)
    manifest["config_hash"] = _detector_artifact_config_hash(manifest)
    manifest["fit_diagnostics"]["artifact_resume_identity"] = manifest[
        "resume_identity"
    ]
    manifest["fit_diagnostics"]["artifact_config_hash"] = manifest[
        "config_hash"
    ]
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _publish_detector_artifact_outputs(output, target_output)
    if staging_dir is not None:
        staging_dir.cleanup()
    output = target_output
    manifest_path = target_output / "manifest.json"

    return DetectorArtifactResult(
        output_dir=output,
        manifest_path=manifest_path,
        shape=shape,
        roi_lower=(roi_x, roi_y),
        roi_dim=(roi_width, roi_height),
        normalization_shift=float(normalization_shift),
        zero_offset_index=int(fit_start),
        zero_offset_status=(
            "manual"
            if zero_result_payload is None
            else str(zero_result_payload.get("status", "cache"))
        ),
        processed_tiles=stats.processed_tiles,
        batched_tiles=stats.batched_tiles,
        raw_fits=stats.raw_fits,
        failures=stats.failures,
        skipped_fits=stats.skipped_fits,
        filtered_fits=stats.filtered_fits,
        elapsed_s=float(elapsed),
    )


def write_detector_artifact_normalization(
    *,
    h5dir: Path | str,
    fon: str,
    foff: str,
    output_dir: Path | str,
    drop_leading: int = 1,
    chunk_frames: int = 16,
    zero_offset: float = 0.0,
    zero_offset_index: int | None = None,
    hdf5_reader: str = "h5py",
    hdf5_reader_workers: int = 2,
) -> DetectorNormalizationResult:
    """Compute and cache detector-wide normalization for sharded runs."""

    _require_nonnegative("drop_leading", drop_leading)
    _require_positive("chunk_frames", chunk_frames)
    _validate_hdf5_reader(hdf5_reader, hdf5_reader_workers)

    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for detector artifacts") from exc

    start_time = perf_counter()
    root = Path(h5dir)
    on_path = _resolve_path(root, fon)
    off_path = _resolve_path(root, foff)
    input_identity = detector_artifact_input_identity(on_path, off_path)
    on_probe = probe_hdf5_file(on_path)
    off_probe = probe_hdf5_file(off_path)
    if on_probe.load_plan is None or off_probe.load_plan is None:
        raise ValueError("unsupported HDF5 schema or IPM layout")
    if on_probe.load_plan.schema != off_probe.load_plan.schema:
        raise ValueError("on/off HDF5 schemas do not match")
    plan = on_probe.load_plan

    with h5py.File(on_path, "r") as on_h5, h5py.File(off_path, "r") as off_h5:
        on_image = on_h5[plan.image_dataset]
        off_image = off_h5[plan.image_dataset]
        if len(on_image.shape) != 3 or len(off_image.shape) != 3:
            raise ValueError(
                "detector artifact input requires 3D image cubes"
            )
        if tuple(on_image.shape) != tuple(off_image.shape):
            raise ValueError("on/off image shapes do not match")
        frame_count, detector_height, detector_width = (
            int(on_image.shape[0]),
            int(on_image.shape[1]),
            int(on_image.shape[2]),
        )
        if drop_leading >= frame_count:
            raise ValueError("drop_leading removes all frames")
        sample_count = frame_count - drop_leading
        delay = np.asarray(
            on_h5[plan.delay_dataset][drop_leading:],
            dtype=np.float64,
        )
        off_delay = np.asarray(
            off_h5[plan.delay_dataset][drop_leading:],
            dtype=np.float64,
        )
        if delay.shape != (sample_count,) or off_delay.shape != delay.shape:
            raise ValueError("delay length does not match image frames")
        if not np.allclose(delay, off_delay):
            raise ValueError("on/off delay axes do not match")
        on_norm = np.asarray(
            on_h5[plan.ipm_pair.normalization][drop_leading:],
            dtype=np.float64,
        )
        off_norm = np.asarray(
            off_h5[plan.ipm_pair.normalization][drop_leading:],
            dtype=np.float64,
        )
        if on_norm.shape != (sample_count,) or off_norm.shape != (
            sample_count,
        ):
            raise ValueError("normalization length does not match frames")

        with _open_hdf5_block_reader(
            mode=hdf5_reader,
            on_path=on_path,
            off_path=off_path,
            image_dataset=plan.image_dataset,
            on_dataset=on_image,
            off_dataset=off_image,
            workers=hdf5_reader_workers,
        ) as block_reader:
            full_trace = _full_detector_trace(
                block_reader,
                on_norm,
                off_norm,
                detector_height=detector_height,
                detector_width=detector_width,
                frame_count=frame_count,
                drop_leading=drop_leading,
                chunk_frames=chunk_frames,
            )

    if zero_offset_index is None:
        zero_result = find_value_drop_position(
            delay,
            full_trace["ratio_minus_one"],
            zero_offset=zero_offset,
        )
        fit_start = int(zero_result.selected_index)
        if fit_start < 0:
            raise ValueError(
                "zero offset could not be found "
                f"({zero_result.status}); pass --zero-offset-index for a "
                "diagnostic or cropped run"
            )
        zero_result_payload = asdict(zero_result)
    else:
        fit_start = int(zero_offset_index)
        zero_result_payload = None
    if fit_start < 0 or fit_start >= sample_count:
        raise ValueError("zero_offset_index is outside usable samples")

    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = tempfile.TemporaryDirectory(
        prefix=f".{output.name}.tmp-",
        dir=output.parent,
    )
    staging = Path(staging_dir.name)
    cache_path = staging / NORMALIZATION_CACHE_FILE
    manifest_path = staging / NORMALIZATION_MANIFEST_FILE
    np.savez(
        cache_path,
        delay=delay,
        on_norm=on_norm,
        off_norm=off_norm,
        ratio_minus_one=full_trace["ratio_minus_one"],
    )
    runtime = runtime_metadata(
        backend="numpy",
        device="cpu",
        dtype=str(on_norm.dtype),
    )
    manifest = {
        "kind": "xray-detector-artifact-normalization",
        "manifest_schema_version": DETECTOR_NORMALIZATION_MANIFEST_VERSION,
        "package_version": runtime["package_version"],
        "backend": "numpy",
        "device": runtime["device"],
        "dtype": runtime["dtype"],
        "runtime": runtime,
        "input_identity": input_identity,
        "normalization_identity": {"kind": "computed-from-input"},
        "schema": plan.schema,
        "image_dataset": plan.image_dataset,
        "delay_dataset": plan.delay_dataset,
        "normalization_dataset": plan.ipm_pair.normalization,
        "drop_leading": int(drop_leading),
        "chunk_frames": int(chunk_frames),
        "zero_offset": float(zero_offset),
        "zero_offset_index": int(fit_start),
        "zero_offset_result": zero_result_payload,
        "normalization_shift": float(full_trace["normalization_shift"]),
        "detector_shape": [int(detector_height), int(detector_width)],
        "frame_count": int(frame_count),
        "sample_count": int(sample_count),
        "hdf5_reader": hdf5_reader,
        "hdf5_reader_runtime": _hdf5_reader_runtime(
            hdf5_reader=hdf5_reader,
            h5py=h5py,
        ),
        "hdf5_reader_workers": int(hdf5_reader_workers),
        "arrays": [NORMALIZATION_CACHE_FILE],
    }
    manifest["config_hash"] = _stable_json_hash(
        {
            key: manifest[key]
            for key in (
                "kind",
                "manifest_schema_version",
                "package_version",
                "backend",
                "dtype",
                "input_identity",
                "normalization_identity",
                "schema",
                "image_dataset",
                "delay_dataset",
                "normalization_dataset",
                "drop_leading",
                "chunk_frames",
                "zero_offset",
                "zero_offset_index",
                "normalization_shift",
                "detector_shape",
                "frame_count",
                "sample_count",
                "hdf5_reader",
                "hdf5_reader_runtime",
                "hdf5_reader_workers",
            )
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output.mkdir(parents=True, exist_ok=True)
    for name in (NORMALIZATION_CACHE_FILE, NORMALIZATION_MANIFEST_FILE):
        (output / name).unlink(missing_ok=True)
        shutil.move(str(staging / name), str(output / name))
    staging_dir.cleanup()

    elapsed = perf_counter() - start_time
    return DetectorNormalizationResult(
        output_dir=output,
        manifest_path=output / NORMALIZATION_MANIFEST_FILE,
        cache_path=output / NORMALIZATION_CACHE_FILE,
        detector_shape=(detector_height, detector_width),
        sample_count=sample_count,
        normalization_shift=float(full_trace["normalization_shift"]),
        zero_offset_index=fit_start,
        zero_offset_status=(
            "manual"
            if zero_result_payload is None
            else str(zero_result_payload.get("status", "cache"))
        ),
        elapsed_s=float(elapsed),
    )


def merge_detector_artifact_shards(
    *,
    shard_dirs: tuple[Path | str, ...],
    output_dir: Path | str,
    strict: bool = True,
    chunk_rows: int = 16,
) -> DetectorArtifactMergeResult:
    """Merge x-sharded detector artifact directories into one artifact."""

    _require_positive("chunk_rows", chunk_rows)
    if not shard_dirs:
        raise ValueError("at least one shard directory is required")
    start_time = perf_counter()
    shards = _load_and_validate_shard_manifests(
        tuple(Path(path) for path in shard_dirs),
        strict=strict,
    )
    first = shards[0]["manifest"]
    global_roi_lower = tuple(
        int(v) for v in first["shard"]["global_roi_lower"]
    )
    global_roi_dim = tuple(int(v) for v in first["shard"]["global_roi_dim"])
    global_width, global_height = global_roi_dim
    depth = int(first["output_shape"][2])
    shape = (global_height, global_width, depth)

    target_output = Path(output_dir)
    target_output.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = tempfile.TemporaryDirectory(
        prefix=f".{target_output.name}.tmp-",
        dir=target_output.parent,
    )
    output = Path(staging_dir.name)
    arrays = {
        "freq_all": _open_output_array(output / "freq_all.npy", shape),
        "amp_all": _open_output_array(output / "amp_all.npy", shape),
        "fft_all": _open_output_array(output / "fft_all.npy", shape),
        "fft_freq_all": _open_output_array(
            output / "fft_freq_all.npy",
            shape,
        ),
        "amp_all_sum_filtered": _open_output_array(
            output / "amp_all_sum_filtered.npy",
            (global_height, global_width),
        ),
        FIT_STATUS_FILE: _open_output_array(
            output / FIT_STATUS_FILE,
            (global_height, global_width),
            dtype=np.uint8,
        ),
    }
    _zero_output_arrays(*arrays.values())

    origin_x, _origin_y = global_roi_lower
    for shard in shards:
        manifest = shard["manifest"]
        shard_path = shard["path"]
        shard_x = int(manifest["roi_lower"][0])
        shard_width = int(manifest["roi_dim"][0])
        output_x = slice(shard_x - origin_x, shard_x - origin_x + shard_width)
        for name in ("freq_all", "amp_all", "fft_all", "fft_freq_all"):
            source = np.load(shard_path / f"{name}.npy", mmap_mode="r")
            target = arrays[name]
            for start in range(0, global_height, chunk_rows):
                stop = min(global_height, start + chunk_rows)
                target[start:stop, output_x, :] = source[start:stop, :, :]
        source_2d = np.load(
            shard_path / "amp_all_sum_filtered.npy",
            mmap_mode="r",
        )
        status_source = np.load(
            shard_path / FIT_STATUS_FILE,
            mmap_mode="r",
        )
        for start in range(0, global_height, chunk_rows):
            stop = min(global_height, start + chunk_rows)
            arrays["amp_all_sum_filtered"][start:stop, output_x] = source_2d[
                start:stop, :
            ]
            arrays[FIT_STATUS_FILE][start:stop, output_x] = status_source[
                start:stop, :
            ]

    for array in arrays.values():
        array.flush()

    fit_diagnostics_metadata = None
    if int(first.get("manifest_schema_version", 1)) >= 2:
        fit_diagnostics_metadata = _merge_fit_diagnostic_shards(
            shards=shards,
            output=output,
            input_identity=first["input_identity"],
        )

    shard_elapsed = float(
        sum(float(item["manifest"].get("elapsed_s", 0.0)) for item in shards)
    )
    merge_elapsed = perf_counter() - start_time
    manifest = dict(first)
    manifest.update(
        {
            "artifact_role": "merged-shards",
            "roi_lower": [int(global_roi_lower[0]), int(global_roi_lower[1])],
            "roi_dim": [int(global_width), int(global_height)],
            "output_shape": [int(value) for value in shape],
            "processed_tiles": int(
                sum(
                    int(item["manifest"].get("processed_tiles", 0))
                    for item in shards
                )
            ),
            "batched_tiles": int(
                sum(
                    int(item["manifest"].get("batched_tiles", 0))
                    for item in shards
                )
            ),
            "raw_fits": int(
                sum(
                    int(item["manifest"].get("raw_fits", 0))
                    for item in shards
                )
            ),
            "failures": int(
                sum(
                    int(item["manifest"].get("failures", 0))
                    for item in shards
                )
            ),
            "skipped_fits": int(
                sum(
                    int(item["manifest"].get("skipped_fits", 0))
                    for item in shards
                )
            ),
            "filtered_fits": int(
                sum(
                    int(item["manifest"].get("filtered_fits", 0))
                    for item in shards
                )
            ),
            "elapsed_s": float(shard_elapsed),
            "merge_elapsed_s": float(merge_elapsed),
            "shard": None,
            "merged_from_shards": [
                {
                    "label": (
                        f"shard-{int(item['manifest']['shard']['index']):04d}"
                    ),
                    "source_path_sha256": sha256(
                        str(item["path"].resolve()).encode("utf-8")
                    ).hexdigest(),
                    "manifest_sha256": sha256(
                        (item["path"] / "manifest.json").read_bytes()
                    ).hexdigest(),
                    "index": int(item["manifest"]["shard"]["index"]),
                    "roi_lower": item["manifest"]["roi_lower"],
                    "roi_dim": item["manifest"]["roi_dim"],
                    "elapsed_s": float(
                        item["manifest"].get("elapsed_s", 0.0)
                    ),
                }
                for item in shards
            ],
            "shard_runtimes": [
                {
                    "index": int(item["manifest"]["shard"]["index"]),
                    "runtime": item["manifest"].get("runtime"),
                }
                for item in shards
            ],
            "shard_count": int(len(shards)),
        }
    )
    if fit_diagnostics_metadata is not None:
        manifest["fit_diagnostics"] = fit_diagnostics_metadata
    manifest["resume_identity"] = detector_artifact_resume_identity(manifest)
    manifest["config_hash"] = _detector_artifact_config_hash(manifest)
    if fit_diagnostics_metadata is not None:
        manifest["fit_diagnostics"]["artifact_resume_identity"] = manifest[
            "resume_identity"
        ]
        manifest["fit_diagnostics"]["artifact_config_hash"] = manifest[
            "config_hash"
        ]
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _publish_detector_artifact_outputs(output, target_output)
    staging_dir.cleanup()

    return DetectorArtifactMergeResult(
        output_dir=target_output,
        manifest_path=target_output / "manifest.json",
        shard_count=len(shards),
        shape=shape,
        roi_lower=global_roi_lower,
        roi_dim=global_roi_dim,
        elapsed_s=float(merge_elapsed),
    )


def _merge_fit_diagnostic_shards(
    *,
    shards: list[dict[str, Any]],
    output: Path,
    input_identity: dict[str, Any],
) -> dict[str, Any]:
    first_metadata = shards[0]["manifest"].get("fit_diagnostics") or {}
    level = _validate_fit_diagnostics(str(first_metadata.get("level")))
    writer = _FitDiagnosticsWriter(
        output,
        level,
        schema_version=int(
            first_metadata.get(
                "schema_version", FIT_DIAGNOSTICS_SCHEMA_VERSION
            )
        ),
    )
    if level != "none":
        for shard in shards:
            metadata = shard["manifest"]["fit_diagnostics"]
            artifact_path = shard["path"] / str(metadata["file"])
            with np.load(artifact_path, allow_pickle=False) as source:
                writer.extend(source)
    return writer.finalize(
        source_identity_sha256=_stable_json_hash(input_identity)
    )


def detector_artifact_complete(
    root: Path | str,
    *,
    expected_resume_identity: str | None = None,
) -> bool:
    """Return whether an artifact is complete with the expected identity."""

    path = Path(root)
    manifest = _read_manifest(path)
    if manifest is None:
        return False
    if (
        expected_resume_identity is not None
        and manifest.get("resume_identity") != expected_resume_identity
    ):
        return False
    if manifest.get("kind") != "xray-detector-artifacts":
        return False
    if manifest.get("manifest_schema_version", 1) not in (
        DETECTOR_ARTIFACT_SUPPORTED_MANIFEST_VERSIONS
    ):
        return False
    try:
        _validate_artifact_files(
            path,
            manifest,
            strict_diagnostics=False,
        )
    except (KeyError, OSError, TypeError, ValueError):
        return False
    return True


def compare_detector_artifacts(
    *,
    reference_dir: Path | str,
    candidate_dir: Path | str,
    amp_threshold: float = 1.6,
    candidate_origin: tuple[int, int] | None = None,
    chunk_rows: int = 16,
) -> dict[str, Any]:
    """Compare two detector artifact directories.

    If the candidate is an ROI artifact with ``manifest.json`` origin
    metadata, the comparison crops the matching region from the reference
    arrays.
    """

    _require_positive("chunk_rows", chunk_rows)
    reference = Path(reference_dir)
    candidate = Path(candidate_dir)
    cand_manifest = _read_manifest(candidate)
    if candidate_origin is None and cand_manifest is not None:
        origin_values = cand_manifest.get("roi_lower")
        if origin_values is not None:
            candidate_origin = (int(origin_values[0]), int(origin_values[1]))
    if candidate_origin is None:
        candidate_origin = (0, 0)

    array_stats: dict[str, Any] = {}
    for name in DETECTOR_ARRAYS:
        left_path = reference / f"{name}.npy"
        right_path = candidate / f"{name}.npy"
        if not left_path.exists() or not right_path.exists():
            array_stats[name] = {
                "present": False,
                "reference_exists": left_path.exists(),
                "candidate_exists": right_path.exists(),
            }
            continue
        left = np.load(left_path, mmap_mode="r")
        right = np.load(right_path, mmap_mode="r")
        array_stats[name] = _compare_arrays(
            left,
            right,
            candidate_origin=candidate_origin,
            chunk_rows=chunk_rows,
        )

    filtered_stats = _compare_filtered_modes(
        reference,
        candidate,
        candidate_origin=candidate_origin,
        amp_threshold=amp_threshold,
        chunk_rows=chunk_rows,
    )
    comparable = all(
        item.get("present", False) and item.get("shape_comparable", False)
        for item in array_stats.values()
    )
    return {
        "kind": "xray-detector-artifact-comparison",
        "reference_dir": str(reference),
        "candidate_dir": str(candidate),
        "candidate_origin": [
            int(candidate_origin[0]),
            int(candidate_origin[1]),
        ],
        "amp_threshold": float(amp_threshold),
        "comparable": bool(comparable),
        "arrays": array_stats,
        "filtered_modes": filtered_stats,
    }


def detector_artifact_origin(root: Path | str) -> tuple[int, int]:
    manifest_path = Path(root) / "manifest.json"
    if not manifest_path.exists():
        return (0, 0)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roi_lower = manifest.get("roi_lower")
    if roi_lower is None:
        return (0, 0)
    if not isinstance(roi_lower, list | tuple) or len(roi_lower) != 2:
        raise ValueError(
            "detector artifact manifest roi_lower must be [x, y]"
        )
    return (int(roi_lower[0]), int(roi_lower[1]))


def detector_artifact_x_index(
    *,
    width: int,
    x_value: int | None,
    origin_x: int,
) -> tuple[int, int]:
    if width < 1:
        raise ValueError("detector artifact width must be positive")
    if x_value is None:
        local_x = width // 2
    else:
        local_x = _detector_artifact_axis_value(
            int(x_value),
            length=width,
            origin=origin_x,
            allow_end=False,
            name="--x-value",
        )
    return local_x, origin_x + local_x


def detector_artifact_y_slice(
    *,
    height: int,
    y_start: int | None,
    y_end: int | None,
    origin_y: int,
) -> tuple[int, int, int, int]:
    if height < 1:
        raise ValueError("detector artifact height must be positive")
    local_start = (
        0
        if y_start is None
        else _detector_artifact_axis_value(
            int(y_start),
            length=height,
            origin=origin_y,
            allow_end=True,
            name="--y-start",
        )
    )
    local_end = (
        height
        if y_end is None
        else _detector_artifact_axis_value(
            int(y_end),
            length=height,
            origin=origin_y,
            allow_end=True,
            name="--y-end",
        )
    )
    if local_end <= local_start:
        raise ValueError(
            "--y-start/--y-end must select a non-empty row range"
        )
    return (
        local_start,
        local_end,
        origin_y + local_start,
        origin_y + local_end,
    )


def _detector_artifact_axis_value(
    value: int,
    *,
    length: int,
    origin: int,
    allow_end: bool,
    name: str,
) -> int:
    upper_delta = length if allow_end else length - 1
    if origin <= value <= origin + upper_delta:
        return value - origin
    if 0 <= value <= upper_delta:
        return value
    raise ValueError(f"{name} is outside detector artifact bounds")


@dataclass
class _RunStats:
    processed_tiles: int = 0
    batched_tiles: int = 0
    raw_fits: int = 0
    failures: int = 0
    skipped_fits: int = 0
    filtered_fits: int = 0


class _FitDiagnosticsWriter:
    """Stream deterministic, pickle-free fit diagnostics into one NPZ."""

    def __init__(
        self,
        output: Path,
        level: FitDiagnosticsLevel,
        *,
        schema_version: int = FIT_DIAGNOSTICS_SCHEMA_VERSION,
    ) -> None:
        _validate_fit_diagnostics(level)
        self._output = output
        self.level = level
        self.schema_version = schema_version
        self.dtypes, self.full_fields, self.ragged_groups = (
            _fit_diagnostic_fields(schema_version)
        )
        self._columns: dict[str, list[int | float | str]] = {
            name: [] for name in self.dtypes
        }
        self._offsets: dict[str, list[int]] = {
            name: [0] for name in self.ragged_groups
        }
        self._raw_paths: dict[str, Path] = {}
        self._streams: dict[str, Any] = {}
        self._time: np.ndarray | None = None
        if level == "full":
            for name in self.full_fields:
                path = output / f".{FIT_DIAGNOSTICS_FILE}.{name}.bin"
                self._raw_paths[name] = path
                self._streams[name] = path.open("wb")

    @property
    def record_count(self) -> int:
        return len(self._columns["detector_y"])

    def append(self, record: dict[str, Any]) -> None:
        if self.level == "none":
            return
        if self.level == "full" and record.get("time") is not None:
            self._set_time(record["time"])
        for name, dtype in self.dtypes.items():
            value = record[name]
            if np.dtype(dtype).kind == "U":
                self._columns[name].append(str(value))
            elif np.issubdtype(dtype, np.integer):
                self._columns[name].append(int(value))
            else:
                self._columns[name].append(_optional_float(value))
        if self.level != "full":
            return

        for offset_name, fields in self.ragged_groups.items():
            values = [
                np.asarray(record[name], dtype=np.float64).reshape(-1)
                for name in fields
            ]
            lengths = {int(value.size) for value in values}
            if len(lengths) != 1:
                raise RuntimeError(
                    f"fit diagnostic {offset_name} arrays differ in length"
                )
            length = lengths.pop()
            for name, value in zip(fields, values, strict=True):
                value.tofile(self._streams[name])
            self._offsets[offset_name].append(
                self._offsets[offset_name][-1] + length
            )

    def extend(self, source: Any) -> None:
        """Append one already-validated NPZ payload during shard merging."""

        if self.level == "none":
            return
        counts = set()
        for name, dtype in self.dtypes.items():
            values = np.asarray(source[name], dtype=dtype).reshape(-1)
            counts.add(int(values.size))
            self._columns[name].extend(values.tolist())
        if len(counts) != 1:
            raise ValueError(
                "fit diagnostic summary columns differ in length"
            )
        record_count = counts.pop()
        if self.level != "full":
            return

        self._set_time(source["time"])

        for offset_name, fields in self.ragged_groups.items():
            offsets = np.asarray(source[offset_name], dtype=np.int64)
            if (
                offsets.shape != (record_count + 1,)
                or int(offsets[0]) != 0
                or np.any(np.diff(offsets) < 0)
            ):
                raise ValueError(
                    f"invalid fit diagnostic offsets: {offset_name}"
                )
            total = int(offsets[-1])
            for name in fields:
                values = np.asarray(source[name], dtype=np.float64).reshape(
                    -1
                )
                if int(values.size) != total:
                    raise ValueError(
                        f"fit diagnostic {name} does not match {offset_name}"
                    )
                values.tofile(self._streams[name])
            base = self._offsets[offset_name][-1]
            self._offsets[offset_name].extend((offsets[1:] + base).tolist())

    def finalize(self, *, source_identity_sha256: str) -> dict[str, Any]:
        if self.level == "none":
            self.close()
            return _fit_diagnostics_manifest_metadata(
                level="none",
                schema_version=self.schema_version,
                source_identity_sha256=source_identity_sha256,
            )

        self.close()
        payload: dict[str, np.ndarray] = {
            "schema_version": np.asarray(self.schema_version, dtype=np.uint16)
        }
        for name, dtype in self.dtypes.items():
            payload[name] = np.asarray(self._columns[name], dtype=dtype)
        if self.level == "full":
            if self._time is None:
                raise RuntimeError(
                    "full fit diagnostics require a common fitted time axis"
                )
            payload["time"] = self._time
            for name, values in self._offsets.items():
                payload[name] = np.asarray(values, dtype=np.int64)
            for name, path in self._raw_paths.items():
                count = int(
                    path.stat().st_size // np.dtype(np.float64).itemsize
                )
                payload[name] = (
                    np.memmap(
                        path, mode="r", dtype=np.float64, shape=(count,)
                    )
                    if count
                    else np.empty((0,), dtype=np.float64)
                )

        artifact_path = self._output / FIT_DIAGNOSTICS_FILE
        array_lengths = {
            name: int(payload[name].size)
            for name in ("time", *self.full_fields)
            if name in payload
        }
        _write_npz_atomic(artifact_path, payload)
        payload.clear()
        for path in self._raw_paths.values():
            path.unlink(missing_ok=True)
        return _fit_diagnostics_manifest_metadata(
            level=self.level,
            schema_version=self.schema_version,
            source_identity_sha256=source_identity_sha256,
            artifact_path=artifact_path,
            record_count=self.record_count,
            array_lengths=array_lengths,
        )

    def close(self) -> None:
        for stream in self._streams.values():
            if not stream.closed:
                stream.close()

    def _set_time(self, value: Any) -> None:
        time = np.asarray(value, dtype=np.float64).reshape(-1)
        if self._time is None:
            self._time = time.copy()
            return
        if self._time.shape != time.shape or not np.array_equal(
            self._time, time
        ):
            raise ValueError(
                "fit diagnostic records do not share a common fitted "
                "time axis"
            )

    def __del__(self) -> None:
        self.close()


def _optional_float(value: Any) -> float:
    return np.nan if value is None else float(value)


def _fit_diagnostics_manifest_metadata(
    *,
    level: FitDiagnosticsLevel,
    source_identity_sha256: str,
    artifact_path: Path | None = None,
    record_count: int = 0,
    array_lengths: dict[str, int] | None = None,
    schema_version: int = FIT_DIAGNOSTICS_SCHEMA_VERSION,
) -> dict[str, Any]:
    _, _, ragged_groups = _fit_diagnostic_fields(schema_version)
    metadata: dict[str, Any] = {
        "level": level,
        "schema_version": schema_version,
        "record_count": int(record_count),
        "record_semantics": (
            "one record per row in every processed detector tile, ordered "
            "by tile x, tile y, then detector y; unprocessed rows have no "
            "record"
        ),
        "non_ok_semantics": (
            "skipped and failed records use NaN float metrics, -1 integer "
            "metrics, and empty full-mode ragged arrays"
        ),
        "coordinate_semantics": (
            "tile bounds are half-open absolute detector pixel coordinates; "
            "detector_y is an absolute detector row"
        ),
        "source_identity_sha256": source_identity_sha256,
        "optional_float_encoding": "IEEE-754 NaN",
        "units": {
            "tile_x_start": "detector pixel index",
            "tile_x_stop": "detector pixel index",
            "tile_y_start": "detector pixel index",
            "tile_y_stop": "detector pixel index",
            "detector_y": "detector pixel index",
            "fit_status": "execution status code",
            "trace_std": "normalized trace units",
            "residual_std": "normalized trace units",
            "relative_residual": "dimensionless",
            "chi2": "squared normalized trace units",
            "selected_model_order": "count",
            "mode_count": "count",
            "p1_rank": "count",
            "p1_singular_value_ratio": "dimensionless",
            "p1_condition": "dimensionless",
            "p2_rank": "count",
            "p2_singular_value_ratio": "dimensionless",
            "p2_condition": "dimensionless",
            "max_amplitude": "normalized trace units",
        },
    }
    if level == "full":
        metadata["units"].update(
            {
                "trace": "normalized trace units",
                "reconstruction": "normalized trace units",
                "time": "delay units",
                "angular_frequency": "radians per delay unit",
                "decay": "inverse delay unit",
                "amplitude": "normalized trace units",
                "phase": "radians",
                "p1_singular_values": "normalized trace units",
                "p2_singular_values": "design-matrix units",
            }
        )
        metadata["ragged_serialization"] = {
            value: key
            for key, fields in ragged_groups.items()
            for value in fields
        }
        metadata["array_lengths"] = dict(array_lengths or {})
        metadata["time_semantics"] = (
            "one common fitted sample axis shared by every nonempty trace "
            "and reconstruction record"
        )
    if schema_version == ITERATIVE_FIT_DIAGNOSTICS_SCHEMA_VERSION:
        units = metadata["units"]
        for name in tuple(units):
            if (
                name.startswith(("p1_", "p2_"))
                or name == "selected_model_order"
            ):
                del units[name]
        units.update(
            {
                "converged": "convergence flag: 1=yes, 0=no, -1=not run",
                "optimizer_status": "optimizer termination reason",
                "iterations": "optimizer iterations",
                "cost": "optimizer objective including amplitude penalty",
                "gradient_norm": "optimizer gradient norm",
                "offset": "normalized trace units",
            }
        )
        metadata["non_ok_semantics"] = (
            "skipped and failed records use NaN float metrics, -1 integer "
            "metrics, and empty full-mode arrays; optimizer termination "
            "fields retain a nonconverged fit's final diagnostics"
        )
    metadata["fit_status_codes"] = {
        "ok": FIT_STATUS_OK,
        "skipped": FIT_STATUS_SKIPPED,
        "failed": FIT_STATUS_FAILED,
    }
    if artifact_path is not None:
        metadata.update(
            {
                "file": artifact_path.name,
                "format": "numpy-npz",
                "allow_pickle": False,
                "artifact_size_bytes": int(artifact_path.stat().st_size),
                "artifact_sha256": _sha256_file(artifact_path),
            }
        )
    else:
        metadata.update(
            {
                "file": None,
                "format": None,
                "allow_pickle": False,
                "artifact_size_bytes": None,
                "artifact_sha256": None,
            }
        )
    return metadata


class _SerialHdf5BlockReader:
    def __init__(self, on_dataset, off_dataset) -> None:
        self._on_dataset = on_dataset
        self._off_dataset = off_dataset

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read_pair(self, selection):
        return (
            np.asarray(self._on_dataset[selection], dtype=np.float64),
            np.asarray(self._off_dataset[selection], dtype=np.float64),
        )


class _ThreadedHdf5BlockReader:
    def __init__(
        self,
        *,
        on_path: Path,
        off_path: Path,
        image_dataset: str,
        workers: int,
    ) -> None:
        self._on_path = on_path
        self._off_path = off_path
        self._image_dataset = image_dataset
        self._executor = ThreadPoolExecutor(max_workers=workers)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._executor.shutdown(wait=True, cancel_futures=True)

    def read_pair(self, selection):
        on_future = self._executor.submit(
            _read_hdf5_block,
            self._on_path,
            self._image_dataset,
            selection,
        )
        off_future = self._executor.submit(
            _read_hdf5_block,
            self._off_path,
            self._image_dataset,
            selection,
        )
        on_result = None
        off_result = None
        first_error = None
        for label, future in (("on", on_future), ("off", off_future)):
            try:
                result = future.result()
            except Exception as exc:
                if first_error is None:
                    first_error = RuntimeError(
                        f"{label} HDF5 block read failed"
                    )
                    first_error.__cause__ = exc
                continue
            if label == "on":
                on_result = result
            else:
                off_result = result
        if first_error is not None:
            raise first_error
        return on_result, off_result


def _open_hdf5_block_reader(
    *,
    mode: str,
    on_path: Path,
    off_path: Path,
    image_dataset: str,
    on_dataset,
    off_dataset,
    workers: int,
):
    if mode == "h5py":
        return _SerialHdf5BlockReader(on_dataset, off_dataset)
    if mode in THREADED_HDF5_READERS:
        return _ThreadedHdf5BlockReader(
            on_path=on_path,
            off_path=off_path,
            image_dataset=image_dataset,
            workers=workers,
        )
    raise ValueError("hdf5_reader must be one of: " + ", ".join(HDF5_READERS))


def _hdf5_reader_runtime(*, hdf5_reader: str, h5py) -> dict[str, Any]:
    config = h5py.get_config()
    runtime = {
        "backend": (
            "h5py" if hdf5_reader == "h5py" else "h5py-worker-threads"
        ),
        "h5py_version": str(getattr(h5py, "__version__", "")),
        "hdf5_version": str(getattr(h5py.version, "hdf5_version", "")),
        "hdf5_built_version": ".".join(
            str(item)
            for item in getattr(h5py.version, "hdf5_built_version_tuple", ())
        ),
        "mpi": bool(getattr(config, "mpi", False)),
        "direct_vfd": bool(getattr(config, "direct_vfd", False)),
        "requested_beta_target": None,
    }
    if hdf5_reader == "hdf5-ts-funcwrap":
        runtime["requested_beta_target"] = {
            "repository": HDF5_TS_FUNCWRAP_REPOSITORY,
            "branch": HDF5_TS_FUNCWRAP_BRANCH,
            "ref": HDF5_TS_FUNCWRAP_REF,
        }
    return runtime


def _read_hdf5_block(path: Path, dataset: str, selection):
    import h5py

    with h5py.File(path, "r") as h5:
        return np.asarray(h5[dataset][selection], dtype=np.float64)


def _fit_error_types(cp) -> tuple[type[BaseException], ...]:
    errors: list[type[BaseException]] = [
        ValueError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ]
    for owner, name in (
        (getattr(cp, "linalg", None), "LinAlgError"),
        (
            getattr(getattr(cp, "cuda", None), "runtime", None),
            "CUDARuntimeError",
        ),
        (
            getattr(getattr(cp, "cuda", None), "memory", None),
            "OutOfMemoryError",
        ),
        (
            getattr(getattr(cp, "cuda", None), "driver", None),
            "CUDADriverError",
        ),
    ):
        error_type = getattr(owner, name, None)
        if error_type is not None:
            errors.append(error_type)
    return tuple(errors)


def _ensure_cuda_device(cp) -> None:
    try:
        device_count = int(cp.cuda.runtime.getDeviceCount())
    except Exception as exc:
        cupy_runtime_error = getattr(
            cp.cuda.runtime,
            "CUDARuntimeError",
            None,
        )
        if cupy_runtime_error is not None and isinstance(
            exc,
            cupy_runtime_error,
        ):
            raise RuntimeError("no CUDA devices visible") from exc
        raise
    if device_count < 1:
        raise RuntimeError("no CUDA devices visible")


def _fit_detector_row(
    *,
    cp,
    time_gpu,
    trace_gpu,
    components: int,
    roots_backend: str,
    padded_length: int,
    fit_method: str = "linear-prediction",
    iterative_options: IterativeFitOptions | None = None,
    p2_ridge_alpha: float = 0.0,
    fit_diagnostics: FitDiagnosticsLevel = "none",
) -> dict[str, Any]:
    if fit_method == "iterative":
        from .iterative_fit import iterative_fit

        result = iterative_fit(
            time_gpu,
            trace_gpu,
            components,
            options=iterative_options,
            backend="gpu",
        )
        if not result.converged:
            raise _IterativeFitConvergenceError(result)
        fft_freq, fft_value = _tdsfft_cupy(cp, time_gpu, trace_gpu)
        row = {
            "freq": _pad_abs(
                result.angular_frequency / (2 * np.pi), padded_length
            ),
            "amp": _pad_abs(result.amplitude, padded_length),
            "fft": _pad_abs(cp.asnumpy(fft_value), padded_length),
            "fft_freq": _pad_abs(cp.asnumpy(fft_freq), padded_length),
        }
        if fit_diagnostics != "none":
            row["diagnostics"] = _iterative_fit_diagnostics(
                cp=cp,
                result=result,
                trace_gpu=trace_gpu,
                level=fit_diagnostics,
            )
        return row
    result = linear_prediction_cupy(
        time_gpu,
        trace_gpu,
        components,
        roots_backend=roots_backend,
        p2_ridge_alpha=p2_ridge_alpha,
        fit_diagnostics=fit_diagnostics != "none",
    )
    frequency_centers = _frequency_centers(
        result.frequency,
        result.spectrum_components,
    )
    fft_freq, fft_value = _tdsfft_cupy(cp, time_gpu, trace_gpu)
    row = {
        "freq": _pad_abs(frequency_centers, padded_length),
        "amp": _pad_abs(result.amplitude, padded_length),
        "fft": _pad_abs(cp.asnumpy(fft_value), padded_length),
        "fft_freq": _pad_abs(cp.asnumpy(fft_freq), padded_length),
    }
    if fit_diagnostics != "none":
        row["diagnostics"] = _linear_prediction_fit_diagnostics(
            cp=cp,
            result=result,
            trace_gpu=trace_gpu,
            level=fit_diagnostics,
        )
    return row


def _fit_detector_rows_batched(
    *,
    cp,
    time_gpu,
    traces_gpu,
    components: int,
    padded_length: int,
) -> tuple[dict[str, np.ndarray] | None, ...]:
    """Batch row artifacts and FFTs without changing P1/P2 solve semantics.

    P1 (SVD and companion roots) and the P2 least-squares solve still run
    one row at a time. A row whose P1 raises a fit error is returned as
    ``None`` so the caller can refit it serially and keep row-local failure
    accounting; the remaining rows stay on the batched artifact path.
    """

    fit_error_types = _fit_error_types(cp)
    row_count = int(traces_gpu.shape[0])
    rows: list[dict[str, np.ndarray] | None] = [None] * row_count
    survivors: list[int] = []
    p1_rows = []
    for row in range(row_count):
        try:
            p1 = _linear_prediction_p1_impl(
                xp=cp,
                time=time_gpu,
                trace=traces_gpu[row],
                n_components=components,
            )
        except fit_error_types:
            continue
        survivors.append(row)
        p1_rows.append(p1)
    if not survivors:
        return tuple(rows)
    if len(survivors) < row_count:
        traces_gpu = traces_gpu[cp.asarray(survivors, dtype=cp.int64)]

    modes = linear_prediction_mode_batch_from_roots_cupy(
        time_gpu,
        tuple(item.eigenvalues for item in p1_rows),
        tuple(item.singular_values for item in p1_rows),
    )
    _filtered, artifacts = linear_prediction_variable_artifacts_cupy_batched(
        time_gpu,
        traces_gpu,
        modes.decay,
        modes.angular_frequency,
        modes.mode_counts,
        solver="serial-lstsq",
    )
    fft_frequency, fft_value = _tdsfft_cupy(cp, time_gpu, traces_gpu)

    output_gpu = cp.zeros(
        (len(survivors), 4, padded_length),
        dtype=cp.float64,
    )
    mode_count = min(
        padded_length,
        int(artifacts.frequency_centers.shape[1]),
    )
    if mode_count:
        output_gpu[:, 0, :mode_count] = cp.abs(
            artifacts.frequency_centers[:, :mode_count]
        )
        output_gpu[:, 1, :mode_count] = cp.abs(
            artifacts.amplitude[:, :mode_count]
        )
    fft_count = min(padded_length, int(fft_value.shape[-1]))
    if fft_count:
        output_gpu[:, 2, :fft_count] = cp.abs(fft_value[:, :fft_count])
        output_gpu[:, 3, :fft_count] = cp.abs(fft_frequency[None, :fft_count])
    output = cp.asnumpy(output_gpu)
    for row, values in zip(survivors, output, strict=True):
        rows[row] = {
            "freq": values[0],
            "amp": values[1],
            "fft": values[2],
            "fft_freq": values[3],
        }
    return tuple(rows)


class _IterativeFitConvergenceError(ValueError):
    def __init__(self, result):
        super().__init__(f"iterative fit did not converge: {result.status}")
        self.iterative_result = result


def _iterative_convergence_diagnostics(result) -> dict[str, Any]:
    return {
        "converged": int(result.converged),
        "optimizer_status": result.status,
        "iterations": result.iterations,
        "cost": result.cost,
        "gradient_norm": result.gradient_norm,
        "offset": result.offset,
    }


def _iterative_fit_diagnostics(*, cp, result, trace_gpu, level):
    payload = {
        "fit_status": FIT_STATUS_OK,
        "trace_std": result.trace_std,
        "residual_std": result.residual_std,
        "relative_residual": result.relative_residual,
        "chi2": result.chi2,
        "mode_count": len(result.amplitude),
        "max_amplitude": (
            float(np.max(result.amplitude)) if len(result.amplitude) else 0.0
        ),
        **_iterative_convergence_diagnostics(result),
    }
    if level == "full":
        payload.update(
            {
                "trace": cp.asnumpy(trace_gpu),
                "reconstruction": result.reconstruction,
                "time": result.time,
                "angular_frequency": result.angular_frequency,
                "decay": result.decay,
                "amplitude": result.amplitude,
                "phase": result.phase,
            }
        )
    return payload


def _linear_prediction_fit_diagnostics(
    *,
    cp,
    result,
    trace_gpu,
    level: FitDiagnosticsLevel,
) -> dict[str, Any]:
    diagnostics = getattr(result, "diagnostics", None)
    if diagnostics is None:
        raise RuntimeError(
            "fit diagnostics were requested, but linear prediction did not "
            "return diagnostic fields"
        )
    payload: dict[str, Any] = {
        "fit_status": FIT_STATUS_OK,
        "trace_std": diagnostics.trace_std,
        "residual_std": diagnostics.residual_std,
        "relative_residual": diagnostics.relative_residual,
        "chi2": diagnostics.chi2,
        "selected_model_order": diagnostics.selected_model_order,
        "mode_count": diagnostics.mode_count,
        "p1_rank": diagnostics.p1_rank,
        "p1_singular_value_ratio": diagnostics.p1_singular_value_ratio,
        "p1_condition": diagnostics.p1_condition,
        "p2_rank": diagnostics.p2_rank,
        "p2_singular_value_ratio": diagnostics.p2_singular_value_ratio,
        "p2_condition": diagnostics.p2_condition,
        "max_amplitude": diagnostics.max_amplitude,
    }
    if level == "full":
        payload.update(
            {
                "trace": cp.asnumpy(trace_gpu),
                "reconstruction": result.reconstruction,
                "time": result.time,
                "angular_frequency": result.angular_frequency,
                "decay": result.decay,
                "amplitude": result.amplitude,
                "phase": result.phase,
                "p1_singular_values": result.singular_values,
                "p2_singular_values": diagnostics.p2_singular_values,
            }
        )
    return payload


def _append_non_ok_fit_diagnostic(
    writer: _FitDiagnosticsWriter,
    *,
    level: FitDiagnosticsLevel,
    fit_status: int,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    detector_y: int,
    iterative_result: Any = None,
) -> None:
    if level == "none":
        return
    payload: dict[str, Any] = {
        name: (
            "not-run"
            if np.dtype(dtype).kind == "U"
            else -1
            if np.issubdtype(dtype, np.integer)
            else None
        )
        for name, dtype in writer.dtypes.items()
    }
    payload.update(
        {
            "tile_x_start": x0,
            "tile_x_stop": x1,
            "tile_y_start": y0,
            "tile_y_stop": y1,
            "detector_y": detector_y,
            "fit_status": fit_status,
        }
    )
    if iterative_result is not None:
        payload.update(_iterative_convergence_diagnostics(iterative_result))
    if level == "full":
        empty = np.empty((0,), dtype=np.float64)
        payload.update({name: empty for name in writer.full_fields})
    writer.append(payload)


def _frequency_centers(
    frequency: np.ndarray,
    spectrum_components: np.ndarray,
) -> np.ndarray:
    frequency = np.asarray(frequency, dtype=np.float64)
    spectrum = np.asarray(spectrum_components, dtype=np.float64)
    if spectrum.ndim != 2:
        raise ValueError("spectrum_components must be two-dimensional")
    if spectrum.shape[1] == 0:
        return np.zeros((0,), dtype=np.float64)
    indices = np.argmax(spectrum, axis=0)
    return frequency[indices]


def _tdsfft_cupy(cp, time, trace):
    if len(time) < 2:
        raise ValueError("FFT requires at least two time samples")
    dt = time[1] - time[0]
    n = int(trace.shape[-1])
    stop = (n + 1) // 2
    frequency = cp.fft.fftfreq(n, d=dt)[:stop]
    value = cp.fft.fft(trace, n, axis=-1)[..., :stop] / n
    return frequency, value


def _pad_abs(values, padded_length: int) -> np.ndarray:
    array = np.abs(np.asarray(values)).astype(np.float64).reshape(-1)
    output = np.zeros((padded_length,), dtype=np.float64)
    count = min(padded_length, int(array.shape[0]))
    if count:
        output[:count] = array[:count]
    return output


def _write_row_outputs(
    freq_all,
    amp_all,
    fft_all,
    fft_freq_all,
    *,
    output_y: int,
    output_x: slice,
    row: dict[str, np.ndarray],
) -> None:
    freq_all[output_y, output_x, :] = row["freq"][None, :]
    amp_all[output_y, output_x, :] = row["amp"][None, :]
    fft_all[output_y, output_x, :] = row["fft"][None, :]
    fft_freq_all[output_y, output_x, :] = row["fft_freq"][None, :]


def _zero_output_arrays(*arrays) -> None:
    if not arrays:
        return
    height = int(arrays[0].shape[0])
    chunk_rows = 16
    for start in range(0, height, chunk_rows):
        stop = min(height, start + chunk_rows)
        for array in arrays:
            array[start:stop, ...] = 0.0


def _integrate_cupy(cp, cp_array, pixels: int):
    """Integrate neighboring detector rows without mixing delay samples."""

    if pixels == 0:
        return cp_array

    samples, rows = cp_array.shape
    horizontal_pixels = min(pixels, max(rows - 1, 0))
    axis1 = cp_array.copy()
    for shift in range(1, horizontal_pixels + 1):
        left = cp.concatenate(
            (
                cp.zeros((samples, shift), dtype=cp_array.dtype),
                cp_array[:, : rows - shift],
            ),
            axis=1,
        )
        axis1 += left
    for shift in range(1, horizontal_pixels + 1):
        right = cp.concatenate(
            (
                cp_array[:, shift:rows],
                cp.zeros((samples, shift), dtype=cp_array.dtype),
            ),
            axis=1,
        )
        axis1 += right
    return axis1


def _row_halo_bounds(
    y0: int,
    y1: int,
    *,
    detector_height: int,
    pixels: int,
) -> tuple[int, int]:
    if pixels <= 0:
        return y0, y1
    return max(0, y0 - pixels), min(detector_height, y1 + pixels)


def _load_tile_signal(
    block_reader,
    on_norm: np.ndarray,
    off_norm: np.ndarray,
    *,
    shift: float,
    drop_leading: int,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
) -> np.ndarray:
    on_block, off_block = block_reader.read_pair(
        (slice(drop_leading, None), slice(y0, y1), slice(x0, x1))
    )
    on_block = np.nan_to_num(on_block, nan=0.0, posinf=0.0, neginf=0.0)
    off_block = np.nan_to_num(off_block, nan=0.0, posinf=0.0, neginf=0.0)
    on_normed = np.divide(
        on_block,
        on_norm[:, None, None],
        out=np.zeros_like(on_block, dtype=np.float64),
        where=on_norm[:, None, None] != 0,
    )
    off_normed = np.divide(
        off_block,
        off_norm[:, None, None],
        out=np.zeros_like(off_block, dtype=np.float64),
        where=off_norm[:, None, None] != 0,
    )
    pixel_offset = float(shift) * int(x1 - x0)
    on_sum = np.sum(on_normed, axis=2) + pixel_offset
    off_sum = np.sum(off_normed, axis=2) + pixel_offset
    signal = np.divide(
        on_sum - off_sum,
        off_sum,
        out=np.zeros_like(on_sum, dtype=np.float64),
        where=off_sum != 0,
    )
    return np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)


def _full_detector_trace(
    block_reader,
    on_norm: np.ndarray,
    off_norm: np.ndarray,
    *,
    detector_height: int,
    detector_width: int,
    frame_count: int,
    drop_leading: int,
    chunk_frames: int,
) -> dict[str, Any]:
    sample_count = frame_count - drop_leading
    on_sums = np.zeros((sample_count,), dtype=np.float64)
    off_sums = np.zeros((sample_count,), dtype=np.float64)
    output_index = 0
    for start in range(drop_leading, frame_count, chunk_frames):
        stop = min(frame_count, start + chunk_frames)
        on_block, off_block = block_reader.read_pair(
            (slice(start, stop), slice(None), slice(None))
        )
        on_block = np.nan_to_num(on_block, nan=0.0, posinf=0.0, neginf=0.0)
        off_block = np.nan_to_num(off_block, nan=0.0, posinf=0.0, neginf=0.0)
        count = stop - start
        on_sums[output_index : output_index + count] = np.sum(
            on_block.reshape(count, -1),
            axis=1,
        )
        off_sums[output_index : output_index + count] = np.sum(
            off_block.reshape(count, -1),
            axis=1,
        )
        output_index += count

    on_normalized = np.divide(
        on_sums,
        on_norm,
        out=np.zeros_like(on_sums, dtype=np.float64),
        where=on_norm != 0,
    )
    off_normalized = np.divide(
        off_sums,
        off_norm,
        out=np.zeros_like(off_sums, dtype=np.float64),
        where=off_norm != 0,
    )
    pixel_count = int(detector_height * detector_width)
    shift = float(np.sum(off_normalized) / (sample_count * pixel_count))
    offset = shift * pixel_count
    ratio = np.divide(
        on_normalized + offset,
        off_normalized + offset,
        out=np.zeros_like(on_normalized, dtype=np.float64),
        where=(off_normalized + offset) != 0,
    )
    return {
        "normalization_shift": shift,
        "ratio_minus_one": ratio - 1.0,
    }


def _write_amp_sum_filtered(
    output_path: Path,
    amp_all,
    *,
    amp_threshold: float,
):
    height, width, _depth = amp_all.shape
    amp_sum = _open_output_array(output_path, (height, width))
    chunk_rows = 16
    for start in range(0, height, chunk_rows):
        stop = min(height, start + chunk_rows)
        block = np.asarray(amp_all[start:stop], dtype=np.float64)
        mask = (block > amp_threshold) & (block < 1e6)
        amp_sum[start:stop, :] = np.sum(np.where(mask, block, 0.0), axis=2)
    return amp_sum


def _compare_arrays(
    reference,
    candidate,
    *,
    candidate_origin: tuple[int, int],
    chunk_rows: int,
) -> dict[str, Any]:
    reference_shape = tuple(int(value) for value in reference.shape)
    candidate_shape = tuple(int(value) for value in candidate.shape)
    crop = _reference_crop(
        reference_shape,
        candidate_shape,
        candidate_origin=candidate_origin,
    )
    stats = {
        "present": True,
        "reference_shape": list(reference_shape),
        "candidate_shape": list(candidate_shape),
        "shape_comparable": crop is not None,
    }
    if crop is None:
        return stats

    max_abs = 0.0
    sum_sq = 0.0
    count = 0
    finite_mismatch_count = 0
    nonfinite_value_mismatch_count = 0
    nonzero_reference = 0
    nonzero_candidate = 0
    nonfinite_reference = 0
    nonfinite_candidate = 0
    for ref_block, cand_block in _iter_comparison_blocks(
        reference,
        candidate,
        crop=crop,
        chunk_rows=chunk_rows,
    ):
        ref = np.asarray(ref_block, dtype=np.float64)
        cand = np.asarray(cand_block, dtype=np.float64)
        if ref.size:
            ref_finite = np.isfinite(ref)
            cand_finite = np.isfinite(cand)
            finite_pair = ref_finite & cand_finite
            if np.any(finite_pair):
                finite_diff = ref[finite_pair] - cand[finite_pair]
                max_abs = max(
                    max_abs,
                    float(np.max(np.abs(finite_diff))),
                )
            finite_mismatch_count += int(
                np.count_nonzero(ref_finite != cand_finite)
            )
            nonfinite_pair = ~ref_finite & ~cand_finite
            if np.any(nonfinite_pair):
                nonfinite_equal = (
                    (np.isnan(ref) & np.isnan(cand))
                    | (np.isposinf(ref) & np.isposinf(cand))
                    | (np.isneginf(ref) & np.isneginf(cand))
                )
                nonfinite_value_mismatch_count += int(
                    np.count_nonzero(nonfinite_pair & ~nonfinite_equal)
                )
            if np.any(finite_pair):
                sum_sq += float(np.sum(finite_diff**2))
            count += int(np.count_nonzero(finite_pair))
            nonfinite_reference += int(np.count_nonzero(~ref_finite))
            nonfinite_candidate += int(np.count_nonzero(~cand_finite))
            nonzero_reference += int(
                np.count_nonzero(ref_finite & (ref != 0))
            )
            nonzero_candidate += int(
                np.count_nonzero(cand_finite & (cand != 0))
            )
    stats.update(
        {
            "elements": int(count),
            "max_abs_diff": float(max_abs),
            "rms_diff": float(np.sqrt(sum_sq / count)) if count else 0.0,
            "finite_mismatch_count": int(finite_mismatch_count),
            "nonfinite_value_mismatch_count": int(
                nonfinite_value_mismatch_count
            ),
            "nonzero_reference": int(nonzero_reference),
            "nonzero_candidate": int(nonzero_candidate),
            "nonfinite_reference": int(nonfinite_reference),
            "nonfinite_candidate": int(nonfinite_candidate),
        }
    )
    return stats


def _compare_filtered_modes(
    reference: Path,
    candidate: Path,
    *,
    candidate_origin: tuple[int, int],
    amp_threshold: float,
    chunk_rows: int,
) -> dict[str, Any]:
    paths = {
        "reference_amp": reference / "amp_all.npy",
        "reference_freq": reference / "freq_all.npy",
        "candidate_amp": candidate / "amp_all.npy",
        "candidate_freq": candidate / "freq_all.npy",
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    if missing:
        return {"present": False, "missing": missing}

    ref_amp = np.load(paths["reference_amp"], mmap_mode="r")
    ref_freq = np.load(paths["reference_freq"], mmap_mode="r")
    cand_amp = np.load(paths["candidate_amp"], mmap_mode="r")
    cand_freq = np.load(paths["candidate_freq"], mmap_mode="r")
    crop = _reference_crop(
        tuple(ref_amp.shape),
        tuple(cand_amp.shape),
        candidate_origin=candidate_origin,
    )
    if crop is None or tuple(ref_freq.shape) != tuple(ref_amp.shape):
        return {"present": True, "shape_comparable": False}
    if tuple(cand_freq.shape) != tuple(cand_amp.shape):
        return {"present": True, "shape_comparable": False}

    total = 0
    agreement = 0
    ref_filtered = 0
    cand_filtered = 0
    intersection = 0
    frequency_max_abs = 0.0
    amplitude_max_abs = 0.0
    for ref_amp_block, cand_amp_block in _iter_comparison_blocks(
        ref_amp,
        cand_amp,
        crop=crop,
        chunk_rows=chunk_rows,
    ):
        # Recompute matching frequency blocks using the same row range.
        start = total // (cand_amp.shape[1] * cand_amp.shape[2])
        rows = cand_amp_block.shape[0]
        stop = start + rows
        ref_freq_block = _slice_reference(ref_freq, crop, start, stop)
        cand_freq_block = cand_freq[start:stop]
        ref_amp_array = np.asarray(ref_amp_block, dtype=np.float64)
        cand_amp_array = np.asarray(cand_amp_block, dtype=np.float64)
        ref_freq_array = np.asarray(ref_freq_block, dtype=np.float64)
        cand_freq_array = np.asarray(cand_freq_block, dtype=np.float64)

        ref_mask = (ref_amp_array > amp_threshold) & (ref_amp_array < 1e6)
        cand_mask = (cand_amp_array > amp_threshold) & (cand_amp_array < 1e6)
        both = ref_mask & cand_mask
        total += int(ref_mask.size)
        agreement += int(np.count_nonzero(ref_mask == cand_mask))
        ref_filtered += int(np.count_nonzero(ref_mask))
        cand_filtered += int(np.count_nonzero(cand_mask))
        intersection += int(np.count_nonzero(both))
        if np.any(both):
            frequency_max_abs = max(
                frequency_max_abs,
                float(
                    np.max(
                        np.abs(ref_freq_array[both] - cand_freq_array[both])
                    )
                ),
            )
            amplitude_max_abs = max(
                amplitude_max_abs,
                float(
                    np.max(np.abs(ref_amp_array[both] - cand_amp_array[both]))
                ),
            )
    return {
        "present": True,
        "shape_comparable": True,
        "elements": int(total),
        "mask_agreement_ratio": float(agreement / total) if total else 1.0,
        "reference_filtered_points": int(ref_filtered),
        "candidate_filtered_points": int(cand_filtered),
        "intersection_filtered_points": int(intersection),
        "frequency_max_abs_diff_on_intersection": float(frequency_max_abs),
        "amplitude_max_abs_diff_on_intersection": float(amplitude_max_abs),
    }


def _iter_comparison_blocks(reference, candidate, *, crop, chunk_rows: int):
    rows = int(candidate.shape[0]) if candidate.ndim >= 2 else 1
    if candidate.ndim < 2:
        yield reference, candidate
        return
    for start in range(0, rows, chunk_rows):
        stop = min(rows, start + chunk_rows)
        yield (
            _slice_reference(reference, crop, start, stop),
            candidate[start:stop],
        )


def _slice_reference(reference, crop, start: int, stop: int):
    if reference.ndim == 2:
        y_slice, x_slice = crop
        return reference[
            slice(y_slice.start + start, y_slice.start + stop),
            x_slice,
        ]
    if reference.ndim == 3:
        y_slice, x_slice, z_slice = crop
        return reference[
            slice(y_slice.start + start, y_slice.start + stop),
            x_slice,
            z_slice,
        ]
    return reference


def _reference_crop(
    reference_shape: tuple[int, ...],
    candidate_shape: tuple[int, ...],
    *,
    candidate_origin: tuple[int, int],
):
    if reference_shape == candidate_shape and candidate_origin == (0, 0):
        if len(reference_shape) == 2:
            return (
                slice(0, reference_shape[0]),
                slice(0, reference_shape[1]),
            )
        if len(reference_shape) == 3:
            return (
                slice(0, reference_shape[0]),
                slice(0, reference_shape[1]),
                slice(0, reference_shape[2]),
            )
    if len(reference_shape) != len(candidate_shape):
        return None
    if len(reference_shape) not in (2, 3):
        return None
    x0, y0 = candidate_origin
    height, width = candidate_shape[:2]
    y1 = y0 + height
    x1 = x0 + width
    if y0 < 0 or x0 < 0 or y1 > reference_shape[0] or x1 > reference_shape[1]:
        return None
    if len(reference_shape) == 3 and reference_shape[2] != candidate_shape[2]:
        return None
    if len(reference_shape) == 2:
        return (slice(y0, y1), slice(x0, x1))
    return (slice(y0, y1), slice(x0, x1), slice(0, candidate_shape[2]))


def _iter_tiles(
    *,
    roi_x: int,
    roi_y: int,
    roi_width: int,
    roi_height: int,
    tile_width: int,
    tile_height: int,
):
    for x0 in range(roi_x, roi_x + roi_width, tile_width):
        x1 = min(x0 + tile_width, roi_x + roi_width)
        for y0 in range(roi_y, roi_y + roi_height, tile_height):
            y1 = min(y0 + tile_height, roi_y + roi_height)
            yield x0, x1, y0, y1


def _resolve_roi(
    *,
    roi_lower: tuple[int, int],
    roi_dim: tuple[int, int] | None,
    detector_width: int,
    detector_height: int,
) -> tuple[int, int, int, int]:
    roi_x, roi_y = int(roi_lower[0]), int(roi_lower[1])
    if roi_x < 0 or roi_y < 0:
        raise ValueError("ROI origin must be non-negative")
    if roi_x >= detector_width or roi_y >= detector_height:
        raise ValueError("ROI origin is outside detector")
    if roi_dim is None:
        roi_width = detector_width - roi_x
        roi_height = detector_height - roi_y
    else:
        roi_width, roi_height = int(roi_dim[0]), int(roi_dim[1])
        if roi_width == 0:
            roi_width = detector_width - roi_x
        if roi_height == 0:
            roi_height = detector_height - roi_y
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("ROI dimensions must be positive")
    if roi_x + roi_width > detector_width:
        raise ValueError("ROI x range exceeds detector width")
    if roi_y + roi_height > detector_height:
        raise ValueError("ROI y range exceeds detector height")
    return roi_x, roi_y, roi_width, roi_height


def _open_output_array(
    path: Path,
    shape: tuple[int, ...],
    *,
    dtype=np.float64,
):
    from numpy.lib.format import open_memmap

    return open_memmap(
        path,
        mode="w+",
        dtype=dtype,
        shape=tuple(int(value) for value in shape),
    )


def _read_manifest(root: Path) -> dict[str, Any] | None:
    path = root / "manifest.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _normalization_cache_paths(
    value: Path | str,
) -> tuple[Path, Path]:
    path = Path(value)
    if path.is_dir():
        return (
            path / NORMALIZATION_MANIFEST_FILE,
            path / NORMALIZATION_CACHE_FILE,
        )
    if path.suffix == ".json":
        return path, path.with_name(NORMALIZATION_CACHE_FILE)
    if path.suffix == ".npz":
        return path.with_name(NORMALIZATION_MANIFEST_FILE), path
    raise ValueError(
        "normalization cache must be a directory, normalization.json, "
        "or normalization.npz"
    )


def _load_detector_normalization_cache(
    value: Path | str,
) -> dict[str, Any]:
    manifest_path, cache_path = _normalization_cache_paths(value)
    if not manifest_path.exists():
        raise ValueError(
            f"normalization manifest does not exist: {manifest_path}"
        )
    if not cache_path.exists():
        raise ValueError(f"normalization cache does not exist: {cache_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(cache_path) as data:
        arrays = {name: np.asarray(data[name]) for name in data.files}
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "cache_path": cache_path,
        "arrays": arrays,
    }


def _validate_detector_normalization_cache(
    payload: dict[str, Any],
    *,
    input_identity: dict[str, Any],
    schema: str,
    image_dataset: str,
    delay_dataset: str,
    normalization_dataset: str,
    detector_height: int,
    detector_width: int,
    frame_count: int,
    sample_count: int,
    drop_leading: int,
    zero_offset: float,
    zero_offset_index: int | None,
    delay: np.ndarray,
    on_norm: np.ndarray,
    off_norm: np.ndarray,
) -> None:
    manifest = payload["manifest"]
    if manifest.get("kind") != "xray-detector-artifact-normalization":
        raise ValueError("normalization cache has an unsupported kind")
    expected = {
        "manifest_schema_version": DETECTOR_NORMALIZATION_MANIFEST_VERSION,
        "package_version": CUPHOTON_VERSION,
        "dtype": "float64",
        "input_identity": input_identity,
        "schema": schema,
        "image_dataset": image_dataset,
        "delay_dataset": delay_dataset,
        "normalization_dataset": normalization_dataset,
        "detector_shape": [int(detector_height), int(detector_width)],
        "frame_count": int(frame_count),
        "sample_count": int(sample_count),
        "drop_leading": int(drop_leading),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(
                f"normalization cache {key} mismatch: "
                f"{manifest.get(key)!r} != {value!r}"
            )
    cache_zero_index = int(manifest.get("zero_offset_index", -1))
    if (
        zero_offset_index is not None
        and cache_zero_index != zero_offset_index
    ):
        raise ValueError(
            "normalization cache zero_offset_index mismatch: "
            f"{cache_zero_index} != {zero_offset_index}"
        )
    if zero_offset_index is None and not np.isclose(
        float(manifest.get("zero_offset", 0.0)),
        float(zero_offset),
    ):
        raise ValueError("normalization cache zero_offset mismatch")
    arrays = payload["arrays"]
    for name, expected_array in (
        ("delay", delay),
        ("on_norm", on_norm),
        ("off_norm", off_norm),
    ):
        if name not in arrays:
            raise ValueError(f"normalization cache is missing {name}")
        if arrays[name].shape != expected_array.shape:
            raise ValueError(f"normalization cache {name} shape mismatch")
        if not np.allclose(arrays[name], expected_array):
            raise ValueError(f"normalization cache {name} values mismatch")


def _load_and_validate_shard_manifests(
    shard_dirs: tuple[Path, ...],
    *,
    strict: bool,
) -> list[dict[str, Any]]:
    shards: list[dict[str, Any]] = []
    for path in shard_dirs:
        manifest = _read_manifest(path)
        if manifest is None:
            raise ValueError(f"missing shard manifest: {path}")
        if manifest.get("kind") != "xray-detector-artifacts":
            raise ValueError(f"unsupported detector artifact shard: {path}")
        shard = manifest.get("shard")
        if not isinstance(shard, dict):
            raise ValueError(f"artifact is missing shard metadata: {path}")
        _validate_detector_artifact_manifest_identity(path, manifest)
        _validate_artifact_files(path, manifest)
        shards.append({"path": path, "manifest": manifest})
    shards.sort(
        key=lambda item: (
            int(item["manifest"]["roi_lower"][0]),
            int(item["manifest"]["shard"]["index"]),
        )
    )
    first = shards[0]["manifest"]
    global_roi_lower = tuple(
        int(v) for v in first["shard"]["global_roi_lower"]
    )
    global_roi_dim = tuple(int(v) for v in first["shard"]["global_roi_dim"])
    expected_x = int(global_roi_lower[0])
    expected_y = int(global_roi_lower[1])
    expected_width = int(global_roi_dim[0])
    expected_height = int(global_roi_dim[1])
    expected_depth = int(first["output_shape"][2])
    expected_count = int(first["shard"].get("count", len(shards)))
    if strict and expected_count != len(shards):
        raise ValueError(
            f"expected {expected_count} shards, received {len(shards)}"
        )
    stable_keys = _detector_artifact_stable_manifest_keys(first)
    seen_indices: set[int] = set()
    for item in shards:
        manifest = item["manifest"]
        shard = manifest["shard"]
        index = int(shard["index"])
        if index in seen_indices:
            raise ValueError(f"duplicate shard index: {index}")
        seen_indices.add(index)
        if tuple(int(v) for v in shard["global_roi_lower"]) != (
            global_roi_lower
        ):
            raise ValueError("shard global_roi_lower mismatch")
        if tuple(int(v) for v in shard["global_roi_dim"]) != global_roi_dim:
            raise ValueError("shard global_roi_dim mismatch")
        if _detector_artifact_stable_manifest_keys(manifest) != stable_keys:
            raise ValueError("shard detector-artifact configuration mismatch")
        roi_x, roi_y = (int(v) for v in manifest["roi_lower"])
        roi_width, roi_height = (int(v) for v in manifest["roi_dim"])
        if roi_y != expected_y or roi_height != expected_height:
            raise ValueError("shards must cover the same full y range")
        if roi_x != expected_x:
            raise ValueError("shard x ranges have a gap or overlap")
        if int(manifest["output_shape"][2]) != expected_depth:
            raise ValueError("shard output depth mismatch")
        expected_x += roi_width
    if expected_x != global_roi_lower[0] + expected_width:
        raise ValueError("shards do not cover the global x range")
    return shards


def _validate_iterative_artifact_configuration(
    manifest: dict[str, Any],
) -> None:
    """Validate complete iterative controls for resume and shard loading."""

    from .iterative_fit import IterativeFitOptions

    if manifest.get("fit_method") != "iterative":
        raise ValueError("manifest v3 requires iterative fitting")
    raw_options = manifest.get("iterative_options")
    if not isinstance(raw_options, dict):
        raise ValueError("manifest is missing iterative options")
    try:
        options = IterativeFitOptions(**raw_options)
        validate_detector_fit_options(
            "iterative", options, manifest["p2_ridge_alpha"]
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid iterative options in manifest") from exc
    if options.to_dict() != raw_options:
        raise ValueError("manifest iterative options must be complete")


def _validate_detector_artifact_manifest_identity(
    path: Path,
    manifest: dict[str, Any],
) -> None:
    required = (
        "manifest_schema_version",
        "package_version",
        "backend",
        "dtype",
        "input_identity",
        "normalization_identity",
        "schema",
        "image_dataset",
        "delay_dataset",
        "normalization_dataset",
        "resume_identity",
        "config_hash",
    )
    missing = [key for key in required if key not in manifest]
    if missing:
        raise ValueError(
            f"shard manifest is missing identity fields at {path.name}: "
            + ", ".join(missing)
        )
    version = manifest["manifest_schema_version"]
    if version not in DETECTOR_ARTIFACT_SUPPORTED_MANIFEST_VERSIONS:
        raise ValueError(
            f"unsupported shard manifest schema at {path.name}: {version!r}"
        )
    if version >= 2:
        fit_diagnostics = manifest.get("fit_diagnostics")
        if not isinstance(fit_diagnostics, dict):
            raise ValueError(
                f"shard manifest is missing fit diagnostics at {path.name}"
            )
        _validate_fit_diagnostics(str(fit_diagnostics.get("level")))
        if "p2_ridge_alpha" not in manifest:
            raise ValueError(
                f"shard manifest is missing p2_ridge_alpha at {path.name}"
            )
        _require_finite_nonnegative(
            "p2_ridge_alpha", manifest["p2_ridge_alpha"]
        )
    if version >= 3:
        _validate_iterative_artifact_configuration(manifest)
    if (
        not isinstance(manifest["package_version"], str)
        or not manifest["package_version"]
    ):
        raise ValueError(f"invalid shard package version at {path.name}")
    for key in ("input_identity", "normalization_identity"):
        if not isinstance(manifest[key], dict) or not manifest[key]:
            raise ValueError(f"invalid shard {key} at {path.name}")
    expected_resume = detector_artifact_resume_identity(manifest)
    if manifest["resume_identity"] != expected_resume:
        raise ValueError(f"shard resume identity mismatch at {path.name}")
    expected_config = _detector_artifact_config_hash(manifest)
    if manifest["config_hash"] != expected_config:
        raise ValueError(f"shard config_hash mismatch at {path.name}")


def _validate_artifact_files(
    path: Path,
    manifest: dict[str, Any],
    *,
    strict_diagnostics: bool = True,
) -> None:
    output_shape = tuple(int(value) for value in manifest["output_shape"])
    roi_dim = tuple(int(value) for value in manifest["roi_dim"])
    expected_2d = (roi_dim[1], roi_dim[0])
    for name in ("freq_all", "amp_all", "fft_all", "fft_freq_all"):
        array_path = path / f"{name}.npy"
        if not array_path.exists():
            raise ValueError(f"missing shard array: {array_path}")
        if tuple(np.load(array_path, mmap_mode="r").shape) != output_shape:
            raise ValueError(f"shard array shape mismatch: {array_path}")
    for name in ("amp_all_sum_filtered.npy", FIT_STATUS_FILE):
        array_path = path / name
        if not array_path.exists():
            raise ValueError(f"missing shard array: {array_path}")
        if tuple(np.load(array_path, mmap_mode="r").shape) != expected_2d:
            raise ValueError(f"shard array shape mismatch: {array_path}")
    _validate_fit_diagnostics_artifact(
        path,
        manifest,
        strict=strict_diagnostics,
    )


def _validate_fit_diagnostics_artifact(
    path: Path,
    manifest: dict[str, Any],
    *,
    strict: bool = True,
) -> None:
    version = int(manifest.get("manifest_schema_version", 1))
    if version == 1:
        return
    if version >= 3:
        _validate_iterative_artifact_configuration(manifest)
    metadata = manifest.get("fit_diagnostics")
    if not isinstance(metadata, dict):
        raise ValueError(f"missing fit diagnostics metadata: {path}")
    level = _validate_fit_diagnostics(str(metadata.get("level")))
    schema_version = (
        ITERATIVE_FIT_DIAGNOSTICS_SCHEMA_VERSION
        if version >= 3
        else FIT_DIAGNOSTICS_SCHEMA_VERSION
    )
    if metadata.get("schema_version") != schema_version:
        raise ValueError(f"fit diagnostics schema mismatch: {path}")
    dtypes, full_fields, ragged_groups = _fit_diagnostic_fields(
        schema_version
    )
    expected_artifact_identity = detector_artifact_resume_identity(manifest)
    if metadata.get("artifact_resume_identity") != expected_artifact_identity:
        raise ValueError(
            f"fit diagnostics artifact resume identity mismatch: {path}"
        )
    expected_config_hash = _detector_artifact_config_hash(manifest)
    if metadata.get("artifact_config_hash") != expected_config_hash:
        raise ValueError(
            f"fit diagnostics artifact config hash mismatch: {path}"
        )
    expected_source = _stable_json_hash(manifest.get("input_identity"))
    if metadata.get("source_identity_sha256") != expected_source:
        raise ValueError(f"fit diagnostics source identity mismatch: {path}")
    if level == "none":
        if metadata.get("file") is not None:
            raise ValueError(
                f"none fit diagnostics unexpectedly declare a file: {path}"
            )
        if int(metadata.get("record_count", -1)) != 0:
            raise ValueError(
                f"none fit diagnostics declare records unexpectedly: {path}"
            )
        return

    filename = metadata.get("file")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
    ):
        raise ValueError(f"invalid fit diagnostics filename: {path}")
    artifact_path = path / filename
    if not artifact_path.exists():
        raise ValueError(f"missing fit diagnostics artifact: {artifact_path}")
    if int(metadata.get("artifact_size_bytes", -1)) != int(
        artifact_path.stat().st_size
    ):
        raise ValueError(f"fit diagnostics size mismatch: {artifact_path}")

    record_count = int(metadata.get("record_count", -1))
    expected_records = sum(
        int(manifest.get(name, -record_count - 1))
        for name in ("raw_fits", "skipped_fits", "failures")
    )
    if record_count != expected_records:
        raise ValueError(
            f"fit diagnostics record count mismatch: {artifact_path}"
        )
    if not strict:
        return
    if metadata.get("artifact_sha256") != _sha256_file(artifact_path):
        raise ValueError(f"fit diagnostics hash mismatch: {artifact_path}")
    with np.load(artifact_path, allow_pickle=False) as data:
        required = {"schema_version", *dtypes}
        if level == "full":
            required.add("time")
            required.update(full_fields)
            required.update(ragged_groups)
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(
                f"fit diagnostics arrays missing at {artifact_path}: "
                + ", ".join(missing)
            )
        if int(np.asarray(data["schema_version"]).item()) != int(
            schema_version
        ):
            raise ValueError(
                f"unsupported fit diagnostics schema: {artifact_path}"
            )
        for name, dtype in dtypes.items():
            values = np.asarray(data[name])
            if values.shape != (record_count,):
                raise ValueError(
                    f"fit diagnostics column shape mismatch: {name}"
                )
            if values.dtype != np.dtype(dtype):
                raise ValueError(
                    f"fit diagnostics column dtype mismatch: {name}"
                )
        statuses = np.asarray(data["fit_status"], dtype=np.uint8)
        allowed_statuses = {
            FIT_STATUS_OK,
            FIT_STATUS_SKIPPED,
            FIT_STATUS_FAILED,
        }
        if not set(np.unique(statuses)).issubset(allowed_statuses):
            raise ValueError(
                "fit diagnostics contain invalid execution status: "
                f"{artifact_path}"
            )
        expected_status_counts = {
            FIT_STATUS_OK: int(manifest.get("raw_fits", -1)),
            FIT_STATUS_SKIPPED: int(manifest.get("skipped_fits", -1)),
            FIT_STATUS_FAILED: int(manifest.get("failures", -1)),
        }
        for status, expected_count in expected_status_counts.items():
            if int(np.count_nonzero(statuses == status)) != expected_count:
                raise ValueError(
                    "fit diagnostics execution status count mismatch: "
                    f"{artifact_path}"
                )
        non_ok = statuses != FIT_STATUS_OK
        if schema_version == ITERATIVE_FIT_DIAGNOSTICS_SCHEMA_VERSION:
            converged = np.asarray(data["converged"])
            reasons = np.asarray(data["optimizer_status"])
            if (
                np.any(converged[~non_ok] != 1)
                or not set(reasons[~non_ok]).issubset(
                    {"converged", "constant"}
                )
                or np.any(converged[non_ok] == 1)
            ):
                raise ValueError(
                    "fit diagnostics convergence status mismatch"
                )
        for name, dtype in dtypes.items():
            if name in {
                "tile_x_start",
                "tile_x_stop",
                "tile_y_start",
                "tile_y_stop",
                "detector_y",
                "fit_status",
            }:
                continue
            if (
                schema_version == ITERATIVE_FIT_DIAGNOSTICS_SCHEMA_VERSION
                and name in _ITERATIVE_CONVERGENCE_DTYPES
            ):
                continue
            values = np.asarray(data[name])
            if np.issubdtype(dtype, np.integer):
                valid_sentinel = np.all(values[non_ok] == -1)
            else:
                valid_sentinel = np.all(np.isnan(values[non_ok]))
            if not valid_sentinel:
                raise ValueError(
                    f"fit diagnostics non-OK sentinel mismatch: {name}"
                )
        if level == "full":
            time = np.asarray(data["time"])
            if time.dtype != np.dtype(np.float64) or time.ndim != 1:
                raise ValueError(
                    f"fit diagnostics time axis invalid: {artifact_path}"
                )
            array_lengths = metadata.get("array_lengths") or {}
            if int(array_lengths.get("time", -1)) != int(time.size):
                raise ValueError(
                    f"fit diagnostics time length mismatch: {artifact_path}"
                )
            for name, fields in ragged_groups.items():
                offsets = np.asarray(data[name])
                if (
                    offsets.dtype != np.dtype(np.int64)
                    or offsets.shape != (record_count + 1,)
                    or int(offsets[0]) != 0
                    or np.any(np.diff(offsets) < 0)
                ):
                    raise ValueError(
                        f"fit diagnostics offsets invalid: {name}"
                    )
                lengths = np.diff(offsets)
                if np.any(lengths[non_ok] != 0):
                    raise ValueError(
                        f"non-OK fit diagnostics contain ragged data: {name}"
                    )
                if name in {"trace_offsets", "reconstruction_offsets"} and (
                    np.any(lengths[~non_ok] != time.size)
                ):
                    raise ValueError(
                        f"fit diagnostics sample length mismatch: {name}"
                    )
                for field in fields:
                    if int(array_lengths.get(field, -1)) != int(offsets[-1]):
                        raise ValueError(
                            f"fit diagnostics array length mismatch: {field}"
                        )


def _detector_artifact_stable_manifest_keys(
    manifest: dict[str, Any],
) -> dict[str, Any]:
    keys = (
        "kind",
        "manifest_schema_version",
        "package_version",
        "backend",
        "dtype",
        "input_identity",
        "normalization_identity",
        "schema",
        "image_dataset",
        "delay_dataset",
        "normalization_dataset",
        "drop_leading",
        "chunk_frames",
        "fit_trailing_drop",
        "zero_offset",
        "requested_zero_offset_index",
        "zero_offset_index",
        "normalization_shift",
        "detector_shape",
        "tile_shape",
        "exclude_y",
        "integrate_pixels",
        "components",
        "roots_backend",
        "savgol_window",
        "savgol_polyorder",
        "amp_threshold",
        "max_fit_failures",
        "hdf5_reader",
        "hdf5_reader_runtime",
        "hdf5_reader_workers",
        "max_tiles",
    )
    payload = {key: manifest.get(key) for key in keys}
    # Omitted or enabled batching retains the pre-opt-out identity.
    if not manifest.get("batch_rows", True):
        payload["batch_rows"] = False
    if int(manifest.get("manifest_schema_version", 1)) >= 2:
        diagnostics = manifest.get("fit_diagnostics") or {}
        payload["fit_diagnostics_level"] = diagnostics.get("level")
        payload["p2_ridge_alpha"] = manifest.get("p2_ridge_alpha")
    if int(manifest.get("manifest_schema_version", 1)) >= 3:
        payload["fit_method"] = manifest.get("fit_method")
        payload["iterative_options"] = manifest.get("iterative_options")
    return payload


def _detector_artifact_config_hash(manifest: dict[str, Any]) -> str:
    return _stable_json_hash(
        _detector_artifact_stable_manifest_keys(manifest)
    )


def _stable_json_hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def detector_artifact_input_identity(
    on_path: Path | str,
    off_path: Path | str,
) -> dict[str, Any]:
    """Return safe, path-redacted identities for an HDF5 input pair."""

    return {
        "on": _safe_file_identity(Path(on_path)),
        "off": _safe_file_identity(Path(off_path)),
    }


def detector_normalization_identity(
    value: Path | str | None,
) -> dict[str, Any]:
    """Return a path-redacted identity for a normalization source."""

    if value is None:
        return {"kind": "computed-from-input"}
    manifest_path, cache_path = _normalization_cache_paths(value)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "kind": "normalization-cache",
        "config_hash": manifest.get("config_hash"),
        "manifest": _safe_file_identity(manifest_path),
        "cache": _safe_file_identity(cache_path),
    }


def detector_artifact_resume_identity(manifest: dict[str, Any]) -> str:
    """Hash the request fields that must match before a shard is resumed."""

    keys = (
        "kind",
        "manifest_schema_version",
        "package_version",
        "backend",
        "dtype",
        "input_identity",
        "normalization_identity",
        "schema",
        "image_dataset",
        "delay_dataset",
        "normalization_dataset",
        "drop_leading",
        "chunk_frames",
        "fit_trailing_drop",
        "zero_offset",
        "requested_zero_offset_index",
        "detector_shape",
        "roi_lower",
        "roi_dim",
        "tile_shape",
        "exclude_y",
        "integrate_pixels",
        "components",
        "roots_backend",
        "savgol_window",
        "savgol_polyorder",
        "amp_threshold",
        "max_fit_failures",
        "hdf5_reader",
        "hdf5_reader_workers",
        "max_tiles",
        "shard",
    )
    payload = {key: manifest.get(key) for key in keys}
    # Omitted or enabled batching retains the pre-opt-out identity.
    if not manifest.get("batch_rows", True):
        payload["batch_rows"] = False
    if int(manifest.get("manifest_schema_version", 1)) >= 2:
        diagnostics = manifest.get("fit_diagnostics") or {}
        payload["fit_diagnostics_level"] = diagnostics.get("level")
        payload["p2_ridge_alpha"] = manifest.get("p2_ridge_alpha")
    if int(manifest.get("manifest_schema_version", 1)) >= 3:
        payload["fit_method"] = manifest.get("fit_method")
        payload["iterative_options"] = manifest.get("iterative_options")
    return _stable_json_hash(payload)


def _safe_file_identity(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve(strict=True)
    before = resolved.stat()
    content_mode, content_sha256 = _file_content_hash(
        resolved, size=int(before.st_size)
    )
    after = resolved.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise RuntimeError(f"input changed while fingerprinting: {path.name}")
    payload = {
        "resolved_path_sha256": sha256(
            str(resolved).encode("utf-8")
        ).hexdigest(),
        "size_bytes": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "content_mode": content_mode,
        "content_sha256": content_sha256,
    }
    payload["identity_sha256"] = _stable_json_hash(payload)
    return payload


def _file_content_hash(path: Path, *, size: int) -> tuple[str, str]:
    digest = sha256()
    with path.open("rb") as stream:
        if size <= _FULL_CONTENT_HASH_LIMIT:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            return "full", digest.hexdigest()
        offsets = (
            0,
            max(0, size // 2 - _CONTENT_SAMPLE_BYTES // 2),
            max(0, size - _CONTENT_SAMPLE_BYTES),
        )
        for offset in offsets:
            stream.seek(offset)
            digest.update(offset.to_bytes(8, "little", signed=False))
            digest.update(stream.read(_CONTENT_SAMPLE_BYTES))
    return "sampled", digest.hexdigest()


def _clear_detector_artifact_outputs(output: Path) -> None:
    for name in DETECTOR_ARRAYS:
        (output / f"{name}.npy").unlink(missing_ok=True)
    (output / FIT_STATUS_FILE).unlink(missing_ok=True)
    (output / FIT_DIAGNOSTICS_FILE).unlink(missing_ok=True)
    (output / "manifest.json").unlink(missing_ok=True)


def _zero_excluded_signal_rows(
    signal: np.ndarray,
    *,
    y0: int,
    y1: int,
    exclude_y: tuple[AxisRange, ...],
) -> None:
    if not exclude_y:
        return
    mask = excluded_row_mask(y0, y1, exclude_y)
    if np.any(mask):
        signal[:, mask] = 0.0


def _publish_detector_artifact_outputs(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    _clear_detector_artifact_outputs(target)
    for name in DETECTOR_ARRAYS:
        shutil.move(
            str(source / f"{name}.npy"),
            str(target / f"{name}.npy"),
        )
    shutil.move(str(source / FIT_STATUS_FILE), str(target / FIT_STATUS_FILE))
    diagnostics_path = source / FIT_DIAGNOSTICS_FILE
    if diagnostics_path.exists():
        shutil.move(str(diagnostics_path), str(target / FIT_DIAGNOSTICS_FILE))
    # The manifest is the completeness marker and is published last.
    shutil.move(str(source / "manifest.json"), str(target / "manifest.json"))


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("wb") as stream:
            np.savez(stream, **arrays)
            stream.flush()
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _validate_hdf5_reader(hdf5_reader: str, hdf5_reader_workers: int) -> None:
    if hdf5_reader not in HDF5_READERS:
        raise ValueError(
            "hdf5_reader must be one of: " + ", ".join(HDF5_READERS)
        )
    if hdf5_reader in THREADED_HDF5_READERS and hdf5_reader_workers < 2:
        raise ValueError(
            "hdf5_reader_workers must be at least 2 for threaded HDF5 readers"
        )


def _validate_fit_diagnostics(level: str) -> FitDiagnosticsLevel:
    if level not in FIT_DIAGNOSTICS_LEVELS:
        raise ValueError(
            "fit_diagnostics must be one of: "
            + ", ".join(FIT_DIAGNOSTICS_LEVELS)
        )
    return cast(FitDiagnosticsLevel, level)


def _require_finite_nonnegative(name: str, value: float) -> None:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{name} must be finite and non-negative")


def _require_positive(name: str, value: int) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")


def _require_nonnegative(name: str, value: int) -> None:
    if int(value) < 0:
        raise ValueError(f"{name} must be non-negative")
