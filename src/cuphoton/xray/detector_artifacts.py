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
from typing import Any

import numpy as np

from cuphoton import __version__ as CUPHOTON_VERSION
from cuphoton.core.runtime import runtime_metadata

from .detector_mask import (
    AxisRange,
    excluded_row_mask,
    format_y_ranges,
)
from .hdf5 import probe_hdf5_file
from .linear_prediction import linear_prediction_cupy
from .zero_offset import find_value_drop_position

DETECTOR_ARRAYS = (
    "freq_all",
    "amp_all",
    "fft_all",
    "fft_freq_all",
    "amp_all_sum_filtered",
)
FIT_STATUS_FILE = "fit_status.npy"
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
DETECTOR_ARTIFACT_MANIFEST_VERSION = 1
_FULL_CONTENT_HASH_LIMIT = 16 * 1024 * 1024
_CONTENT_SAMPLE_BYTES = 64 * 1024


@dataclass(frozen=True)
class DetectorArtifactResult:
    """Completed detector-wide artifact build.

    ``shape`` is ``(roi_y, roi_x, spectral_bins)`` for emitted detector
    arrays, ``roi_lower`` and ``roi_dim`` use ``(x, y)`` ordering in detector
    pixels, and ``elapsed_s`` is wall time in seconds. Output paths own the
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
    roots_backend: str = "eigvals",
    savgol_window: int = 5,
    savgol_polyorder: int = 3,
    amp_threshold: float = 1.6,
    max_fit_failures: int = 0,
    hdf5_reader: str = "h5py",
    hdf5_reader_workers: int = 2,
    max_tiles: int | None = None,
    normalization_cache: Path | str | None = None,
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
    """

    _require_positive("tile width", tile_shape[0])
    _require_positive("tile height", tile_shape[1])
    _require_nonnegative("drop_leading", drop_leading)
    _require_positive("chunk_frames", chunk_frames)
    _require_nonnegative("fit_trailing_drop", fit_trailing_drop)
    _require_nonnegative("integrate_pixels", integrate_pixels)
    _require_positive("components", components)
    _require_positive("savgol_window", savgol_window)
    _require_nonnegative("savgol_polyorder", savgol_polyorder)
    _require_nonnegative("max_fit_failures", max_fit_failures)
    _require_positive("hdf5_reader_workers", hdf5_reader_workers)
    _validate_hdf5_reader(hdf5_reader, hdf5_reader_workers)
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
                for local_row, skip in enumerate(skip_rows):
                    output_y = y0 + local_row - roi_y
                    output_x = slice(x0 - roi_x, x1 - roi_x)
                    if skip:
                        fit_status[output_y, output_x] = FIT_STATUS_SKIPPED
                        stats.skipped_fits += 1
                        continue
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
                            roots_backend=roots_backend,
                            padded_length=padded_length,
                        )
                    except fit_error_types as exc:
                        fit_status[output_y, output_x] = FIT_STATUS_FAILED
                        stats.failures += 1
                        if stats.failures > max_fit_failures:
                            raise RuntimeError(
                                "detector artifact generation had "
                                f"{stats.failures} fit failures; allowed "
                                f"{max_fit_failures}"
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

    elapsed = perf_counter() - start_time
    runtime = runtime_metadata(backend="cupy", dtype="float64")
    manifest = {
        "kind": "xray-detector-artifacts",
        "manifest_schema_version": DETECTOR_ARTIFACT_MANIFEST_VERSION,
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
        "roots_backend": roots_backend,
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
        "raw_fits": int(stats.raw_fits),
        "failures": int(stats.failures),
        "skipped_fits": int(stats.skipped_fits),
        "filtered_fits": int(stats.filtered_fits),
        "elapsed_s": float(elapsed),
        "arrays": [f"{name}.npy" for name in DETECTOR_ARRAYS]
        + [FIT_STATUS_FILE],
        "fit_status_codes": {
            "unprocessed": FIT_STATUS_UNPROCESSED,
            "ok": FIT_STATUS_OK,
            "skipped": FIT_STATUS_SKIPPED,
            "failed": FIT_STATUS_FAILED,
        },
    }
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
        "manifest_schema_version": DETECTOR_ARTIFACT_MANIFEST_VERSION,
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
    manifest["resume_identity"] = detector_artifact_resume_identity(manifest)
    manifest["config_hash"] = _detector_artifact_config_hash(manifest)
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
    output_shape = manifest.get("output_shape")
    roi_dim = manifest.get("roi_dim")
    if output_shape is None or roi_dim is None:
        return False
    expected_3d = tuple(int(value) for value in output_shape)
    expected_2d = (int(roi_dim[1]), int(roi_dim[0]))
    for name in ("freq_all", "amp_all", "fft_all", "fft_freq_all"):
        array_path = path / f"{name}.npy"
        if not array_path.exists():
            return False
        if tuple(np.load(array_path, mmap_mode="r").shape) != expected_3d:
            return False
    for name in ("amp_all_sum_filtered.npy", FIT_STATUS_FILE):
        array_path = path / name
        if not array_path.exists():
            return False
        if tuple(np.load(array_path, mmap_mode="r").shape) != expected_2d:
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
    raw_fits: int = 0
    failures: int = 0
    skipped_fits: int = 0
    filtered_fits: int = 0


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
) -> dict[str, np.ndarray]:
    result = linear_prediction_cupy(
        time_gpu,
        trace_gpu,
        components,
        roots_backend=roots_backend,
    )
    frequency_centers = _frequency_centers(
        result.frequency,
        result.spectrum_components,
    )
    fft_freq, fft_value = _tdsfft_cupy(cp, time_gpu, trace_gpu)
    return {
        "freq": _pad_abs(frequency_centers, padded_length),
        "amp": _pad_abs(result.amplitude, padded_length),
        "fft": _pad_abs(cp.asnumpy(fft_value), padded_length),
        "fft_freq": _pad_abs(cp.asnumpy(fft_freq), padded_length),
    }


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
    n = len(trace)
    stop = (n + 1) // 2
    frequency = cp.fft.fftfreq(n, d=dt)[:stop]
    value = cp.fft.fft(trace, n)[:stop] / n
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
        "manifest_schema_version": DETECTOR_ARTIFACT_MANIFEST_VERSION,
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
    if (
        manifest["manifest_schema_version"]
        != DETECTOR_ARTIFACT_MANIFEST_VERSION
    ):
        raise ValueError(
            f"unsupported shard manifest schema at {path.name}: "
            f"{manifest['manifest_schema_version']!r}"
        )
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


def _validate_artifact_files(path: Path, manifest: dict[str, Any]) -> None:
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
    return {key: manifest.get(key) for key in keys}


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
    return _stable_json_hash({key: manifest.get(key) for key in keys})


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
    for name in (FIT_STATUS_FILE, "manifest.json"):
        shutil.move(str(source / name), str(target / name))


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


def _require_positive(name: str, value: int) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be positive")


def _require_nonnegative(name: str, value: int) -> None:
    if int(value) < 0:
        raise ValueError(f"{name} must be non-negative")
