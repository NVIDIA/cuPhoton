# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Iterable as IterableABC
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .detector_mask import AxisRange
from .detector_mask import excluded_row_mask as detector_excluded_row_mask


@dataclass(frozen=True)
class IpmPair:
    label: str
    primary: str
    secondary: str
    normalization: str


@dataclass(frozen=True)
class Hdf5LoadPlan:
    schema: str
    image_dataset: str
    delay_dataset: str
    entries_dataset: str
    ipm_pair: IpmPair


@dataclass(frozen=True)
class DatasetProbe:
    name: str
    shape: tuple[int, ...]
    dtype: str
    chunks: tuple[int, ...] | None


@dataclass(frozen=True)
class FileProbe:
    path: Path
    size_bytes: int
    keys: tuple[str, ...]
    datasets: tuple[DatasetProbe, ...]
    schema: str
    ipm_pairs: tuple[str, ...]
    load_plan: Hdf5LoadPlan | None


@dataclass(frozen=True)
class Hdf5PairProbe:
    on: FileProbe
    off: FileProbe

    def to_dict(self) -> dict[str, Any]:
        return {
            "on": _file_probe_to_dict(self.on),
            "off": _file_probe_to_dict(self.off),
        }


def probe_hdf5_pair(
    *,
    h5dir: Path | str,
    fon: str,
    foff: str,
) -> Hdf5PairProbe:
    root = Path(h5dir)
    return Hdf5PairProbe(
        on=probe_hdf5_file(_resolve_hdf5_path(root, fon)),
        off=probe_hdf5_file(_resolve_hdf5_path(root, foff)),
    )


def probe_hdf5_file(path: Path | str) -> FileProbe:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for HDF5 probing") from exc

    h5_path = Path(path)
    if not h5_path.exists():
        raise FileNotFoundError(h5_path)
    if not h5_path.is_file():
        raise ValueError(f"not a file: {h5_path}")

    with h5py.File(h5_path, "r") as h5:
        keys = tuple(sorted(h5.keys()))
        datasets = []
        for key in keys:
            obj = h5[key]
            if hasattr(obj, "shape"):
                chunks = None if obj.chunks is None else tuple(obj.chunks)
                datasets.append(
                    DatasetProbe(
                        name=key,
                        shape=tuple(int(dim) for dim in obj.shape),
                        dtype=str(obj.dtype),
                        chunks=chunks,
                    )
                )
    return FileProbe(
        path=h5_path,
        size_bytes=h5_path.stat().st_size,
        keys=keys,
        datasets=tuple(datasets),
        schema=classify_hdf5_schema(keys),
        ipm_pairs=detect_ipm_pairs(keys),
        load_plan=build_hdf5_load_plan(keys),
    )


def classify_hdf5_schema(keys: tuple[str, ...] | set[str]) -> str:
    key_set = set(keys)
    if {"imgs", "scan_var", "i0", "bin_count", "ROI"} <= key_set:
        return "cropped-cube"
    if {"jungfrau1M_data", "binVar_bins", "nEntries"} <= key_set:
        return "legate-cube"
    return "unknown"


IPM_PAIRS = (
    IpmPair(
        label="ipm3/ipm2",
        primary="ipm3__sum",
        secondary="ipm2__sum",
        normalization="ipm2__sum",
    ),
    IpmPair(
        label="ipm5/ipm4",
        primary="ipm5__sum",
        secondary="ipm4__sum",
        normalization="ipm4__sum",
    ),
    IpmPair(
        label="i0/i0_ipm3",
        primary="i0",
        secondary="i0_ipm3",
        normalization="i0",
    ),
)


def detect_ipm_pairs(keys: tuple[str, ...] | set[str]) -> tuple[str, ...]:
    return tuple(pair.label for pair in _available_ipm_pairs(keys))


def select_ipm_pair(
    keys: tuple[str, ...] | set[str],
    *,
    schema: str | None = None,
) -> IpmPair:
    """Select the IPM pair used by the reference analysis."""

    candidates = _available_ipm_pairs(keys)
    if schema == "cropped-cube":
        candidates = tuple(
            pair for pair in candidates if pair.label == "i0/i0_ipm3"
        )
    elif schema == "legate-cube":
        candidates = tuple(
            pair
            for pair in candidates
            if pair.label in {"ipm3/ipm2", "ipm5/ipm4"}
        )
    if not candidates:
        raise ValueError("no supported IPM pair found")
    return candidates[0]


def build_hdf5_load_plan(
    keys: tuple[str, ...] | set[str],
) -> Hdf5LoadPlan | None:
    schema = classify_hdf5_schema(keys)
    if schema == "cropped-cube":
        try:
            ipm_pair = select_ipm_pair(keys, schema=schema)
        except ValueError:
            return None
        return Hdf5LoadPlan(
            schema=schema,
            image_dataset="imgs",
            delay_dataset="scan_var",
            entries_dataset="bin_count",
            ipm_pair=ipm_pair,
        )
    if schema == "legate-cube":
        try:
            ipm_pair = select_ipm_pair(keys, schema=schema)
        except ValueError:
            return None
        return Hdf5LoadPlan(
            schema=schema,
            image_dataset="jungfrau1M_data",
            delay_dataset="binVar_bins",
            entries_dataset="nEntries",
            ipm_pair=ipm_pair,
        )
    return None


@dataclass(frozen=True)
class Hdf5CubeTrace:
    """Frame-reduced trace from one detector cube.

    Attributes
    ----------
    path, schema
        Source file and recognized input schema.
    image_dataset, delay_dataset, normalization_dataset
        HDF5 dataset names selected by the load plan.
    delay
        One-dimensional scan coordinates in source-file units.
    normalized_sum
        Per-frame detector or ROI sums divided by ``normalization``.
    normalization
        One-dimensional incident-monitor values in source-file units.
    pixel_count
        Number of detector pixels contributing to each sum.

    Notes
    -----
    Arrays are owned NumPy values. The HDF5 file is closed before this result
    is returned.
    """

    path: Path
    schema: str
    image_dataset: str
    delay_dataset: str
    normalization_dataset: str
    delay: np.ndarray
    normalized_sum: np.ndarray
    normalization: np.ndarray
    pixel_count: int


@dataclass(frozen=True)
class Hdf5PairTrace:
    """Aligned on/off traces and their dimensionless normalized ratio.

    Attributes
    ----------
    on, off
        Equal-length reduced traces with matching delay coordinates.
    shift
        Per-pixel reference offset in normalized-signal units.
    ratio_minus_one
        One-dimensional ``(on + offset) / (off + offset) - 1`` trace.
    """

    on: Hdf5CubeTrace
    off: Hdf5CubeTrace
    shift: float
    ratio_minus_one: np.ndarray

    @property
    def delay(self) -> np.ndarray:
        return self.on.delay

    def summary(self) -> dict[str, Any]:
        return {
            "samples": int(self.ratio_minus_one.shape[0]),
            "delay_min": float(np.min(self.delay)),
            "delay_max": float(np.max(self.delay)),
            "ratio_min": float(np.min(self.ratio_minus_one)),
            "ratio_mean": float(np.mean(self.ratio_minus_one)),
            "ratio_max": float(np.max(self.ratio_minus_one)),
            "shift": self.shift,
            "on": _trace_side_summary(self.on),
            "off": _trace_side_summary(self.off),
        }


@dataclass(frozen=True)
class RoiCandidate:
    x: int
    y: int
    width: int
    height: int
    score: float
    usable_rows: int
    row_peak_max: float
    row_std_mean: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_hdf5_pair_trace(
    *,
    h5dir: Path | str,
    fon: str,
    foff: str,
    drop_leading: int = 1,
    chunk_frames: int = 16,
    reference_shift: bool = True,
) -> Hdf5PairTrace:
    """Reduce an on/off cube pair to the reference full-detector trace."""

    root = Path(h5dir)
    on = load_hdf5_cube_trace(
        _resolve_hdf5_path(root, fon),
        drop_leading=drop_leading,
        chunk_frames=chunk_frames,
    )
    off = load_hdf5_cube_trace(
        _resolve_hdf5_path(root, foff),
        drop_leading=drop_leading,
        chunk_frames=chunk_frames,
    )
    if on.schema != off.schema:
        raise ValueError(f"schema mismatch: on={on.schema} off={off.schema}")
    if on.delay.shape != off.delay.shape or not np.allclose(
        on.delay, off.delay
    ):
        raise ValueError("on/off delay axes do not match")
    if on.pixel_count != off.pixel_count:
        raise ValueError("on/off image pixel counts do not match")

    shift = 0.0
    if reference_shift:
        shift = float(np.sum(off.normalized_sum))
        shift /= float(off.normalized_sum.shape[0] * off.pixel_count)
    offset = shift * float(on.pixel_count)
    denominator = off.normalized_sum + offset
    ratio = np.divide(
        on.normalized_sum + offset,
        denominator,
        out=np.zeros_like(on.normalized_sum, dtype=np.float64),
        where=denominator != 0,
    )
    return Hdf5PairTrace(
        on=on,
        off=off,
        shift=shift,
        ratio_minus_one=ratio - 1.0,
    )


def load_hdf5_pair_roi_trace(
    *,
    h5dir: Path | str,
    fon: str,
    foff: str,
    roi_x: int,
    roi_y: int,
    roi_width: int,
    roi_height: int,
    row_y: int | None = None,
    exclude_y: IterableABC[AxisRange] = (),
    drop_leading: int = 1,
    chunk_frames: int = 16,
    reference_shift: bool = True,
) -> Hdf5PairTrace:
    """Reduce an on/off cube pair to an ROI or detector-row trace."""

    root = Path(h5dir)
    exclude_y = tuple(exclude_y)
    on = load_hdf5_cube_roi_trace(
        _resolve_hdf5_path(root, fon),
        roi_x=roi_x,
        roi_y=roi_y,
        roi_width=roi_width,
        roi_height=roi_height,
        row_y=row_y,
        exclude_y=exclude_y,
        drop_leading=drop_leading,
        chunk_frames=chunk_frames,
    )
    off = load_hdf5_cube_roi_trace(
        _resolve_hdf5_path(root, foff),
        roi_x=roi_x,
        roi_y=roi_y,
        roi_width=roi_width,
        roi_height=roi_height,
        row_y=row_y,
        exclude_y=exclude_y,
        drop_leading=drop_leading,
        chunk_frames=chunk_frames,
    )
    if on.schema != off.schema:
        raise ValueError(f"schema mismatch: on={on.schema} off={off.schema}")
    if on.delay.shape != off.delay.shape or not np.allclose(
        on.delay, off.delay
    ):
        raise ValueError("on/off delay axes do not match")
    if on.pixel_count != off.pixel_count:
        raise ValueError("on/off ROI pixel counts do not match")

    shift = 0.0
    if reference_shift:
        shift = float(np.sum(off.normalized_sum))
        shift /= float(off.normalized_sum.shape[0] * off.pixel_count)
    offset = shift * float(on.pixel_count)
    denominator = off.normalized_sum + offset
    ratio = np.divide(
        on.normalized_sum + offset,
        denominator,
        out=np.zeros_like(on.normalized_sum, dtype=np.float64),
        where=denominator != 0,
    )
    return Hdf5PairTrace(
        on=on,
        off=off,
        shift=shift,
        ratio_minus_one=ratio - 1.0,
    )


def scan_roi_candidates(
    *,
    h5dir: Path | str,
    fon: str,
    foff: str,
    tile_width: int = 16,
    tile_height: int = 16,
    stride_x: int = 128,
    stride_y: int = 128,
    drop_leading: int = 1,
    max_candidates: int = 10,
    min_row_std: float = 1e-9,
    exclude_y: IterableABC[AxisRange] = (),
) -> list[RoiCandidate]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError(
            "h5py is required for ROI candidate scanning"
        ) from exc

    if tile_width <= 0 or tile_height <= 0:
        raise ValueError("tile dimensions must be positive")
    if stride_x <= 0 or stride_y <= 0:
        raise ValueError("strides must be positive")
    if drop_leading < 0:
        raise ValueError("drop_leading must be non-negative")
    if max_candidates <= 0:
        raise ValueError("max_candidates must be positive")
    exclude_y = tuple(exclude_y)

    root = Path(h5dir)
    on_path = _resolve_hdf5_path(root, fon)
    off_path = _resolve_hdf5_path(root, foff)
    on_probe = probe_hdf5_file(on_path)
    off_probe = probe_hdf5_file(off_path)
    if on_probe.load_plan is None or off_probe.load_plan is None:
        raise ValueError("unsupported HDF5 schema or IPM layout")
    if on_probe.load_plan.schema != off_probe.load_plan.schema:
        raise ValueError("on/off HDF5 schemas do not match")
    plan = on_probe.load_plan

    candidates: list[RoiCandidate] = []
    with h5py.File(on_path, "r") as on_h5, h5py.File(off_path, "r") as off_h5:
        on_image = on_h5[plan.image_dataset]
        off_image = off_h5[plan.image_dataset]
        if len(on_image.shape) != 3 or len(off_image.shape) != 3:
            raise ValueError("ROI candidate scan requires 3D image cubes")
        if tuple(on_image.shape) != tuple(off_image.shape):
            raise ValueError("on/off image shapes do not match")
        frame_count, detector_height, detector_width = (
            int(on_image.shape[0]),
            int(on_image.shape[1]),
            int(on_image.shape[2]),
        )
        if drop_leading >= frame_count:
            raise ValueError("drop_leading removes all frames")
        if tile_width > detector_width or tile_height > detector_height:
            raise ValueError("tile dimensions exceed detector dimensions")

        on_norm = np.asarray(
            on_h5[plan.ipm_pair.normalization][drop_leading:],
            dtype=np.float64,
        )
        off_norm = np.asarray(
            off_h5[plan.ipm_pair.normalization][drop_leading:],
            dtype=np.float64,
        )

        for y in range(0, detector_height - tile_height + 1, stride_y):
            for x in range(0, detector_width - tile_width + 1, stride_x):
                score = _score_roi_candidate(
                    on_image,
                    off_image,
                    on_norm,
                    off_norm,
                    x=x,
                    y=y,
                    width=tile_width,
                    height=tile_height,
                    drop_leading=drop_leading,
                    min_row_std=min_row_std,
                    exclude_y=exclude_y,
                )
                candidates.append(score)

    candidates.sort(
        key=lambda item: (item.usable_rows, item.score, item.row_peak_max),
        reverse=True,
    )
    return candidates[:max_candidates]


def load_hdf5_cube_trace(
    path: Path | str,
    *,
    drop_leading: int = 1,
    chunk_frames: int = 16,
) -> Hdf5CubeTrace:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for HDF5 trace loading") from exc

    if drop_leading < 0:
        raise ValueError("drop_leading must be non-negative")
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")

    h5_path = Path(path)
    probe = probe_hdf5_file(h5_path)
    if probe.load_plan is None:
        raise ValueError(f"unsupported HDF5 schema or IPM layout: {h5_path}")
    plan = probe.load_plan

    with h5py.File(h5_path, "r") as h5:
        image = h5[plan.image_dataset]
        if len(image.shape) < 2:
            raise ValueError(
                f"image dataset is not at least 2D: {image.name}"
            )
        frame_count = int(image.shape[0])
        if drop_leading >= frame_count:
            raise ValueError("drop_leading removes all frames")
        delay = np.asarray(h5[plan.delay_dataset][drop_leading:], dtype=float)
        normalization = np.asarray(
            h5[plan.ipm_pair.normalization][drop_leading:],
            dtype=float,
        )
        expected = frame_count - drop_leading
        if delay.shape != (expected,):
            raise ValueError(
                f"delay length {delay.shape} does not match {expected}"
            )
        if normalization.shape != (expected,):
            raise ValueError(
                "normalization length "
                f"{normalization.shape} does not match {expected}"
            )

        normalized_sum = _normalized_frame_sums(
            image,
            normalization,
            drop_leading=drop_leading,
            chunk_frames=chunk_frames,
        )
        pixel_count = int(np.prod(image.shape[1:]))

    return Hdf5CubeTrace(
        path=h5_path,
        schema=plan.schema,
        image_dataset=plan.image_dataset,
        delay_dataset=plan.delay_dataset,
        normalization_dataset=plan.ipm_pair.normalization,
        delay=delay,
        normalized_sum=normalized_sum,
        normalization=normalization,
        pixel_count=pixel_count,
    )


def load_hdf5_cube_roi_trace(
    path: Path | str,
    *,
    roi_x: int,
    roi_y: int,
    roi_width: int,
    roi_height: int,
    row_y: int | None = None,
    exclude_y: IterableABC[AxisRange] = (),
    drop_leading: int = 1,
    chunk_frames: int = 16,
) -> Hdf5CubeTrace:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required for HDF5 trace loading") from exc

    if drop_leading < 0:
        raise ValueError("drop_leading must be non-negative")
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    if roi_x < 0 or roi_y < 0:
        raise ValueError("ROI origin must be non-negative")
    if roi_width <= 0 or roi_height <= 0:
        raise ValueError("ROI dimensions must be positive")

    h5_path = Path(path)
    probe = probe_hdf5_file(h5_path)
    if probe.load_plan is None:
        raise ValueError(f"unsupported HDF5 schema or IPM layout: {h5_path}")
    plan = probe.load_plan

    with h5py.File(h5_path, "r") as h5:
        image = h5[plan.image_dataset]
        if len(image.shape) != 3:
            raise ValueError("ROI trace loading requires a 3D image cube")
        frame_count, detector_height, detector_width = (
            int(image.shape[0]),
            int(image.shape[1]),
            int(image.shape[2]),
        )
        if drop_leading >= frame_count:
            raise ValueError("drop_leading removes all frames")
        if roi_x + roi_width > detector_width:
            raise ValueError("ROI x range exceeds detector width")
        if roi_y + roi_height > detector_height:
            raise ValueError("ROI y range exceeds detector height")
        delay = np.asarray(h5[plan.delay_dataset][drop_leading:], dtype=float)
        normalization = np.asarray(
            h5[plan.ipm_pair.normalization][drop_leading:],
            dtype=float,
        )
        expected = frame_count - drop_leading
        if delay.shape != (expected,):
            raise ValueError(
                f"delay length {delay.shape} does not match {expected}"
            )
        if normalization.shape != (expected,):
            raise ValueError(
                "normalization length "
                f"{normalization.shape} does not match {expected}"
            )
        row_mask = _roi_row_mask(
            roi_y=roi_y,
            roi_height=roi_height,
            row_y=row_y,
            exclude_y=exclude_y,
        )
        pixel_count = int(np.count_nonzero(row_mask) * roi_width)
        if pixel_count <= 0:
            raise ValueError("ROI row selection has no usable pixels")
        normalized_sum = _normalized_roi_sums(
            image,
            normalization,
            drop_leading=drop_leading,
            chunk_frames=chunk_frames,
            roi_x=roi_x,
            roi_y=roi_y,
            roi_width=roi_width,
            roi_height=roi_height,
            row_mask=row_mask,
        )

    return Hdf5CubeTrace(
        path=h5_path,
        schema=plan.schema,
        image_dataset=plan.image_dataset,
        delay_dataset=plan.delay_dataset,
        normalization_dataset=plan.ipm_pair.normalization,
        delay=delay,
        normalized_sum=normalized_sum,
        normalization=normalization,
        pixel_count=pixel_count,
    )


def _score_roi_candidate(
    on_image,
    off_image,
    on_norm: np.ndarray,
    off_norm: np.ndarray,
    *,
    x: int,
    y: int,
    width: int,
    height: int,
    drop_leading: int,
    min_row_std: float,
    exclude_y: IterableABC[AxisRange] = (),
) -> RoiCandidate:
    on_roi = np.asarray(
        on_image[drop_leading:, y : y + height, x : x + width],
        dtype=np.float64,
    )
    off_roi = np.asarray(
        off_image[drop_leading:, y : y + height, x : x + width],
        dtype=np.float64,
    )
    on_roi = np.nan_to_num(on_roi, nan=0.0, posinf=0.0, neginf=0.0)
    off_roi = np.nan_to_num(off_roi, nan=0.0, posinf=0.0, neginf=0.0)

    on_normed = np.divide(
        on_roi,
        on_norm[:, None, None],
        out=np.zeros_like(on_roi, dtype=np.float64),
        where=on_norm[:, None, None] != 0,
    )
    off_normed = np.divide(
        off_roi,
        off_norm[:, None, None],
        out=np.zeros_like(off_roi, dtype=np.float64),
        where=off_norm[:, None, None] != 0,
    )
    shift = float(np.mean(off_normed))
    on_normed += shift
    off_normed += shift

    s2_on = np.sum(on_normed, axis=2)
    s2_off = np.sum(off_normed, axis=2)
    signal = np.divide(
        s2_on - s2_off,
        s2_off,
        out=np.zeros_like(s2_on, dtype=np.float64),
        where=s2_off != 0,
    )
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    row_std = np.std(signal, axis=0)
    row_peak = np.max(np.abs(signal), axis=0)
    allowed_rows = np.logical_not(
        np.asarray(
            detector_excluded_row_mask(y, y + height, exclude_y),
            dtype=bool,
        )
    )
    usable = (row_std > min_row_std) & allowed_rows
    usable_rows = int(np.count_nonzero(usable))
    score = float(np.sum(row_peak[usable])) if usable_rows else 0.0
    allowed_peak = row_peak[allowed_rows]
    allowed_std = row_std[allowed_rows]
    return RoiCandidate(
        x=int(x),
        y=int(y),
        width=int(width),
        height=int(height),
        score=score,
        usable_rows=usable_rows,
        row_peak_max=(
            float(np.max(allowed_peak)) if allowed_peak.shape[0] else 0.0
        ),
        row_std_mean=(
            float(np.mean(allowed_std)) if allowed_std.shape[0] else 0.0
        ),
    )


def _resolve_hdf5_path(root: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute():
        return path
    return root / path


def _available_ipm_pairs(
    keys: tuple[str, ...] | set[str],
) -> tuple[IpmPair, ...]:
    key_set = set(keys)
    return tuple(
        pair
        for pair in IPM_PAIRS
        if {pair.primary, pair.secondary} <= key_set
    )


def _normalized_frame_sums(
    image,
    normalization: np.ndarray,
    *,
    drop_leading: int,
    chunk_frames: int,
) -> np.ndarray:
    normalized_sum = np.zeros(normalization.shape[0], dtype=np.float64)
    out_index = 0
    frame_count = int(image.shape[0])
    for start in range(drop_leading, frame_count, chunk_frames):
        stop = min(frame_count, start + chunk_frames)
        frames = np.asarray(image[start:stop], dtype=np.float64)
        frames = np.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)
        frame_sums = np.sum(frames.reshape(frames.shape[0], -1), axis=1)
        norm = normalization[out_index : out_index + frame_sums.shape[0]]
        normalized_sum[out_index : out_index + frame_sums.shape[0]] = (
            np.divide(
                frame_sums,
                norm,
                out=np.zeros(frame_sums.shape, dtype=np.float64),
                where=norm != 0,
            )
        )
        out_index += frame_sums.shape[0]
    return normalized_sum


def _normalized_roi_sums(
    image,
    normalization: np.ndarray,
    *,
    drop_leading: int,
    chunk_frames: int,
    roi_x: int,
    roi_y: int,
    roi_width: int,
    roi_height: int,
    row_mask: np.ndarray,
) -> np.ndarray:
    normalized_sum = np.zeros(normalization.shape[0], dtype=np.float64)
    out_index = 0
    frame_count = int(image.shape[0])
    for start in range(drop_leading, frame_count, chunk_frames):
        stop = min(frame_count, start + chunk_frames)
        row_slice = slice(roi_y, roi_y + roi_height)
        col_slice = slice(roi_x, roi_x + roi_width)
        frames = np.asarray(
            image[start:stop, row_slice, col_slice],
            dtype=np.float64,
        )
        frames = np.nan_to_num(frames, nan=0.0, posinf=0.0, neginf=0.0)
        frames = frames[:, row_mask, :]
        frame_sums = np.sum(frames.reshape(frames.shape[0], -1), axis=1)
        norm = normalization[out_index : out_index + frame_sums.shape[0]]
        normalized_sum[out_index : out_index + frame_sums.shape[0]] = (
            np.divide(
                frame_sums,
                norm,
                out=np.zeros(frame_sums.shape, dtype=np.float64),
                where=norm != 0,
            )
        )
        out_index += frame_sums.shape[0]
    return normalized_sum


def _roi_row_mask(
    *,
    roi_y: int,
    roi_height: int,
    row_y: int | None,
    exclude_y: IterableABC[AxisRange],
) -> np.ndarray:
    excluded = np.asarray(
        detector_excluded_row_mask(roi_y, roi_y + roi_height, exclude_y),
        dtype=bool,
    )
    allowed = np.logical_not(excluded)
    if row_y is None:
        return allowed
    if row_y < roi_y or row_y >= roi_y + roi_height:
        raise ValueError("row_y must fall inside the ROI y range")
    relative = row_y - roi_y
    if not allowed[relative]:
        raise ValueError("row_y is excluded by detector mask")
    row_mask = np.zeros(roi_height, dtype=bool)
    row_mask[relative] = True
    return row_mask


def _trace_side_summary(trace: Hdf5CubeTrace) -> dict[str, Any]:
    return {
        "path": str(trace.path),
        "schema": trace.schema,
        "image_dataset": trace.image_dataset,
        "delay_dataset": trace.delay_dataset,
        "normalization_dataset": trace.normalization_dataset,
        "pixel_count": trace.pixel_count,
        "normalized_sum_min": float(np.min(trace.normalized_sum)),
        "normalized_sum_mean": float(np.mean(trace.normalized_sum)),
        "normalized_sum_max": float(np.max(trace.normalized_sum)),
    }


def _file_probe_to_dict(probe: FileProbe) -> dict[str, Any]:
    payload = asdict(probe)
    payload["path"] = str(probe.path)
    return payload
