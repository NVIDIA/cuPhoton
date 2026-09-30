# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Lossless tile-row storage for detector spectra and bounded slice reads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Never

import numpy as np

SPECTRAL_ARRAYS = ("freq_all", "amp_all", "fft_all", "fft_freq_all")
SPECTRAL_LAYOUT_FILE = "spectral-layout.json"
ARTIFACT_LAYOUTS = ("dense", "tile-rows")


class TileRowArray:
    """Read logical detector pixels from one stored spectrum per x tile.

    Integer and slice indexing expands only the requested pixels. The backing
    spectra are memory mapped; no dense detector cube is allocated on open.
    """

    ndim = 3

    def __init__(
        self,
        values: np.ndarray,
        shape: tuple[int, int, int],
        x_edges: np.ndarray,
    ) -> None:
        self.values = values
        self.shape = shape
        self.x_edges = x_edges
        self.dtype = values.dtype

    def __len__(self) -> int:
        return self.shape[0]

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> Never:
        raise TypeError(
            "index a bounded slice before converting detector spectra to a "
            "NumPy array; array[:] expands the full logical cube"
        )

    def __getitem__(self, key: Any) -> np.ndarray:
        indices = key if isinstance(key, tuple) else (key,)
        if any(value is Ellipsis for value in indices):
            if sum(value is Ellipsis for value in indices) != 1:
                raise IndexError("only one ellipsis is supported")
            expanded = []
            for value in indices:
                expanded.extend(
                    [slice(None)] * (4 - len(indices))
                    if value is Ellipsis
                    else [value]
                )
            indices = tuple(expanded)
        if len(indices) > 3 or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer, slice))
            for value in indices
        ):
            raise IndexError(
                "detector spectra support integer and slice indexing, "
                "not boolean or advanced indexing"
            )
        y, x, z = indices + (slice(None),) * (3 - len(indices))
        if isinstance(x, slice):
            columns = np.arange(*x.indices(self.shape[1]), dtype=np.int64)
        else:
            if not -self.shape[1] <= x < self.shape[1]:
                raise IndexError("detector x index is out of range")
            columns = np.asarray(x % self.shape[1])
        tile_indices = np.searchsorted(
            self.x_edges[1:], columns, side="right"
        )
        block = self.values[y, :, z]
        return np.take(block, tile_indices, axis=int(isinstance(y, slice)))


DetectorArray = np.ndarray | TileRowArray


def spectral_path(path: Path, layout: str) -> Path:
    """Return the physical filename for a logical spectral NPY path."""
    if layout == "dense":
        return path
    if layout != "tile-rows":
        raise ValueError("artifact_layout must be dense or tile-rows")
    return path.with_suffix(".tile-rows.npy")


def detector_array_exists(path: Path) -> bool:
    return path.exists() or (
        path.stem in SPECTRAL_ARRAYS
        and spectral_path(path, "tile-rows").exists()
        and (path.parent / SPECTRAL_LAYOUT_FILE).exists()
    )


def read_spectral_layout(
    root: Path,
) -> tuple[tuple[int, int, int], np.ndarray]:
    payload = json.loads((root / SPECTRAL_LAYOUT_FILE).read_text())
    if (
        not isinstance(payload, dict)
        or payload.get("schema") != "cuphoton.xray.tile-rows/v1"
    ):
        raise ValueError("unsupported detector spectral layout")
    shape = payload.get("shape")
    edges = payload.get("x_edges")
    return _validate_spectral_shape_edges(shape, edges)


def _validate_spectral_shape_edges(
    shape: Any, edges: Any
) -> tuple[tuple[int, int, int], np.ndarray]:
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(type(value) is not int or value <= 0 for value in shape)
        or not isinstance(edges, list)
        or len(edges) < 2
        or any(type(value) is not int for value in edges)
        or edges[0] != 0
        or edges[-1] != shape[1]
        or any(
            left >= right
            for left, right in zip(edges, edges[1:], strict=False)
        )
    ):
        raise ValueError("invalid detector spectral shape or x edges")
    return (shape[0], shape[1], shape[2]), np.asarray(edges, dtype=np.int64)


def load_detector_array(path: Path | str) -> DetectorArray:
    """Open dense or compact detector spectra with logical pixel indexing."""
    path = Path(path)
    if path.exists() or path.stem not in SPECTRAL_ARRAYS:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    shape, edges = read_spectral_layout(path.parent)
    values = np.load(
        spectral_path(path, "tile-rows"), mmap_mode="r", allow_pickle=False
    )
    if values.shape != (shape[0], len(edges) - 1, shape[2]):
        raise ValueError("tile-row array shape differs from spectral layout")
    if values.dtype != np.dtype(np.float64):
        raise ValueError("tile-row detector spectra require float64 values")
    return TileRowArray(values, shape, edges)


def create_detector_spectra(
    root: Path,
    shape: tuple[int, int, int],
    *,
    x_edges: np.ndarray | None = None,
) -> dict[str, np.memmap]:
    """Create writable spectra, optionally storing each x tile only once."""
    layout = "dense" if x_edges is None else "tile-rows"
    physical_shape = shape
    if x_edges is not None:
        shape, x_edges = _validate_spectral_shape_edges(
            list(shape), x_edges.tolist()
        )
        physical_shape = (shape[0], len(x_edges) - 1, shape[2])
        payload = {
            "schema": "cuphoton.xray.tile-rows/v1",
            "shape": list(shape),
            "x_edges": x_edges.tolist(),
        }
        (root / SPECTRAL_LAYOUT_FILE).write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )
    return {
        name: np.lib.format.open_memmap(
            spectral_path(root / f"{name}.npy", layout),
            mode="w+",
            dtype=np.float64,
            shape=physical_shape,
        )
        for name in SPECTRAL_ARRAYS
    }
