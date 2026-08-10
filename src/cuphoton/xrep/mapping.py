# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""WCS mapping and coordinate-grid helpers for xRep."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from astropy.wcs import WCS
from astropy.wcs.wcs import NoConvergence

from .geometry import BBox, Grid, ReprojectionSpec
from .io import load_fits_image_with_wcs


@dataclass(slots=True)
class PreparedReprojection:
    """Prepared per-pixel source lookup arrays for a reprojection.

    Attributes
    ----------
    x_local, y_local
        Zero-based source coordinates with shape ``output_shape``. Coordinates
        are local to the provided source array after ``source_origin`` is
        subtracted.
    relative_area
        Absolute local Jacobian determinant
        ``|d(source pixel)/d(destination pixel)|``. Multiplying sampled
        per-pixel flux by this value accounts for the pixel-area change.
    valid
        Boolean array with ``output_shape``; true means both coordinates are
        finite and inside the source array. Interpolation footprints can apply
        additional edge rules.
    """

    x_local: np.ndarray
    y_local: np.ndarray
    relative_area: np.ndarray
    valid: np.ndarray

    @property
    def output_shape(self) -> tuple[int, int]:
        """Return destination ``(height, width)``."""

        return self.x_local.shape


def _high_level_pixel_to_pixel(
    from_wcs: Any,
    to_wcs: Any,
    coords: np.ndarray,
) -> np.ndarray:
    """Transform pixels through unit-aware WCS APIs when available.

    The high-level API carries world-coordinate units and object types between
    WCS implementations. The low-level fallback preserves compatibility with
    existing duck-typed WCS objects that only implement Astropy's legacy
    numeric methods.
    """

    points = np.asarray(coords, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("coords must have shape (N, 2)")

    if not (
        callable(getattr(from_wcs, "pixel_to_world", None))
        and callable(getattr(to_wcs, "world_to_pixel", None))
    ):
        world = from_wcs.all_pix2world(points, 0)
        return np.asarray(
            to_wcs.all_world2pix(world, 0, quiet=True),
            dtype=np.float64,
        )

    world = from_wcs.pixel_to_world(points[:, 0], points[:, 1])
    world_args = (
        tuple(world) if isinstance(world, (list, tuple)) else (world,)
    )
    try:
        pixels = to_wcs.world_to_pixel(*world_args)
    except NoConvergence as exc:
        best = np.asarray(exc.best_solution, dtype=np.float64)
        if best.shape != points.shape:
            raise
        return best

    if not isinstance(pixels, (list, tuple)) or len(pixels) != 2:
        raise ValueError("two-dimensional WCS must return two pixel arrays")
    return np.column_stack(
        [
            np.asarray(pixels[0], dtype=np.float64),
            np.asarray(pixels[1], dtype=np.float64),
        ]
    )


def make_wcs_pair_transform(from_wcs: WCS, to_wcs: WCS):
    """Compose two WCS objects into a pixel-to-pixel mapping.

    Parameters
    ----------
    from_wcs
        WCS interpreting input pixel coordinates.
    to_wcs
        WCS producing output pixel coordinates.

    Returns
    -------
    callable
        Vectorized ``(N, 2)`` pixel-coordinate transform.
    """

    def mapping(coords: np.ndarray) -> np.ndarray:
        result = _high_level_pixel_to_pixel(from_wcs, to_wcs, coords)
        bad = ~np.isfinite(result).all(axis=1, keepdims=True)
        return np.where(bad, -1.0, result)

    return mapping


def make_wcs_pair_transform_with_offset(
    from_wcs: WCS,
    to_wcs: WCS,
    *,
    offset: tuple[int, int],
):
    """Compose two WCS objects after translating local input coordinates.

    Parameters
    ----------
    from_wcs, to_wcs
        Input and output world-coordinate systems.
    offset
        Integer ``(x, y)`` offset added to input pixels.

    Returns
    -------
    callable
        Vectorized ``(N, 2)`` pixel-coordinate transform.
    """

    delta = np.array([[offset[0], offset[1]]], dtype=np.float64)

    def mapping(coords: np.ndarray) -> np.ndarray:
        result = _high_level_pixel_to_pixel(
            from_wcs,
            to_wcs,
            coords + delta,
        )
        bad = ~np.isfinite(result).all(axis=1, keepdims=True)
        return np.where(bad, -1.0, result)

    return mapping


def make_grid_mapping(
    source_wcs: WCS,
    *,
    grid: Grid,
    output_bbox: BBox,
):
    """Build a destination-to-source mapping onto a shared grid.

    Parameters
    ----------
    source_wcs
        Source-image WCS.
    grid
        Destination shared grid.
    output_bbox
        Local output bounding box on the grid.

    Returns
    -------
    callable
        Vectorized destination-to-source coordinate transform.
    """

    return make_wcs_pair_transform_with_offset(
        grid.wcs,
        source_wcs,
        offset=output_bbox.origin,
    )


def make_wcs_mapping(
    fits_path: Path,
    grid: Grid,
    *,
    hdu: int | None = None,
    output_bbox: BBox | None = None,
) -> tuple[Any, BBox]:
    """Build a FITS-backed destination-to-source mapping.

    Parameters
    ----------
    fits_path
        Source FITS image.
    grid
        Destination shared grid.
    hdu
        Optional explicit two-dimensional image HDU.
    output_bbox
        Optional destination footprint; derived from the source when omitted.

    Returns
    -------
    mapping
        Vectorized destination-to-source transform.
    bbox
        Destination footprint used by the mapping.
    """

    image, source_wcs, _, _ = load_fits_image_with_wcs(fits_path, hdu=hdu)
    bbox = output_bbox or estimate_source_bbox_on_grid(
        source_wcs,
        shape=image.shape,
        grid=grid,
    )
    return make_grid_mapping(source_wcs, grid=grid, output_bbox=bbox), bbox


def estimate_source_bbox_on_grid(
    source_wcs: WCS,
    *,
    shape: tuple[int, int],
    grid: Grid,
    perimeter_step: int = 16,
) -> BBox:
    """Estimate a source-image footprint on a shared grid.

    Parameters
    ----------
    source_wcs
        Source-image WCS.
    shape
        Source ``(height, width)``.
    grid
        Destination shared grid.
    perimeter_step
        Sampling interval along source-image edges.

    Returns
    -------
    BBox
        Integer bounding box containing the sampled footprint.
    """

    perimeter = _perimeter_samples(shape[0], shape[1], step=perimeter_step)
    perimeter_grid = _high_level_pixel_to_pixel(
        source_wcs,
        grid.wcs,
        perimeter,
    )
    finite = np.all(np.isfinite(perimeter_grid), axis=1)
    if not np.any(finite):
        raise ValueError(
            "source WCS perimeter has no finite projection onto the grid"
        )
    perimeter_grid = perimeter_grid[finite]
    x_min = int(np.floor(perimeter_grid[:, 0].min()))
    y_min = int(np.floor(perimeter_grid[:, 1].min()))
    x_max = int(np.ceil(perimeter_grid[:, 0].max()))
    y_max = int(np.ceil(perimeter_grid[:, 1].max()))
    return BBox(
        min_x=x_min,
        min_y=y_min,
        width=x_max - x_min + 1,
        height=y_max - y_min + 1,
    )


def prepare_reprojection(
    spec: ReprojectionSpec,
    *,
    source_shape: tuple[int, int],
    xp=np,
) -> PreparedReprojection:
    """Prepare source-coordinate and area arrays for one reproject.

    Parameters
    ----------
    spec
        Backend-neutral reprojection specification.
    source_shape
        Source ``(height, width)``.
    xp
        NumPy-compatible array module used for dense expansion.

    Returns
    -------
    PreparedReprojection
        Dense source coordinates, area scaling, and valid footprint.

    Notes
    -----
    The WCS mapping is evaluated on a coarse NumPy grid because Astropy is
    CPU-only. Only the full-resolution expansion uses ``xp``.
    """

    if len(source_shape) != 2:
        raise ValueError(f"Expected a 2D source shape, saw {source_shape!r}")
    height, width = source_shape
    if spec.mapping_grid_step < 1:
        raise ValueError("mapping_grid_step must be >= 1")
    if spec.output_bbox.width <= 0 or spec.output_bbox.height <= 0:
        raise ValueError("output bbox dimensions must be positive")

    source_ext = interpolate_mapping_grid(
        spec.mapping,
        width=spec.output_bbox.width,
        height=spec.output_bbox.height,
        interp_len=spec.mapping_grid_step,
        xp=xp,
    )

    sx_ext = source_ext[..., 0]
    sy_ext = source_ext[..., 1]

    sx_cur = sx_ext[1:, 1:]
    sy_cur = sy_ext[1:, 1:]
    sx_pl = sx_ext[:-1, :-1]
    sy_pl = sy_ext[:-1, :-1]
    sx_pu = sx_ext[:-1, 1:]
    sy_pu = sy_ext[:-1, 1:]

    d_a_x = sx_cur - sx_pl
    d_a_y = sy_cur - sy_pl
    d_b_x = sx_cur - sx_pu
    d_b_y = sy_cur - sy_pu
    relative_area = xp.asarray(
        xp.abs(d_a_x * d_b_y - d_a_y * d_b_x),
        dtype=xp.float64,
    )

    x_local = sx_cur - float(spec.source_origin[0])
    y_local = sy_cur - float(spec.source_origin[1])
    valid = (
        xp.isfinite(x_local)
        & xp.isfinite(y_local)
        & (x_local >= 0.0)
        & (x_local <= float(width - 1))
        & (y_local >= 0.0)
        & (y_local <= float(height - 1))
    )
    relative_area = xp.where(
        valid & xp.isfinite(relative_area),
        relative_area,
        0.0,
    )
    return PreparedReprojection(
        x_local=xp.asarray(x_local, dtype=xp.float64),
        y_local=xp.asarray(y_local, dtype=xp.float64),
        relative_area=relative_area,
        valid=valid,
    )


def interpolate_mapping_grid(
    map_fn,
    *,
    width: int,
    height: int,
    interp_len: int,
    xp=np,
) -> np.ndarray:
    """Evaluate a mapping on a coarse grid and bilinearly interpolate it.

    The mapping is evaluated on the coarse grid with NumPy/astropy on the CPU;
    the dense expansion is performed with ``xp`` (NumPy or CuPy).
    """

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if interp_len < 1:
        raise ValueError("interp_len must be >= 1")

    max_col = width - 1
    max_row = height - 1

    # Coarse edges + mapping evaluation stay on the CPU (astropy needs numpy).
    x_edges_np = np.arange(-1, max_col, interp_len, dtype=np.int64)
    if x_edges_np[-1] != max_col:
        x_edges_np = np.concatenate([x_edges_np, [max_col]])
    y_edges_np = np.arange(-1, max_row, interp_len, dtype=np.int64)
    if y_edges_np[-1] != max_row:
        y_edges_np = np.concatenate([y_edges_np, [max_row]])

    xx, yy = np.meshgrid(x_edges_np, y_edges_np, indexing="xy")
    edge_pts = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float64)
    mapped_np = np.asarray(map_fn(edge_pts), dtype=np.float64)
    if mapped_np.shape != edge_pts.shape:
        raise ValueError("map_fn must return an (N, 2) array")
    mapped_np = mapped_np.reshape(y_edges_np.size, x_edges_np.size, 2)

    # Dense expansion uses xp (may be CuPy).
    mapped = xp.asarray(mapped_np)
    x_edges = xp.asarray(x_edges_np)
    y_edges = xp.asarray(y_edges_np)

    xs = xp.arange(-1, width, dtype=xp.int64)
    ys = xp.arange(-1, height, dtype=xp.int64)
    j = xp.searchsorted(x_edges[1:], xs, side="left") + 1
    i = xp.searchsorted(y_edges[1:], ys, side="left") + 1
    jg = xp.broadcast_to(j[None, :], (height + 1, width + 1))
    ig = xp.broadcast_to(i[:, None], (height + 1, width + 1))

    # Cast edges/lattice to float64 before dividing: NumPy/CuPy promote
    # int/int to float64, but torch promotes to float32 — which would lose
    # sub-pixel precision in the fractions below.
    x_prev = xp.asarray(x_edges[jg - 1], dtype=xp.float64)
    x_next = xp.asarray(x_edges[jg], dtype=xp.float64)
    y_prev = xp.asarray(y_edges[ig - 1], dtype=xp.float64)
    y_next = xp.asarray(y_edges[ig], dtype=xp.float64)
    xs_f = xp.asarray(xs, dtype=xp.float64)
    ys_f = xp.asarray(ys, dtype=xp.float64)

    tx = ((xs_f[None, :] - x_prev) / (x_next - x_prev))[..., None]
    ty = ((ys_f[:, None] - y_prev) / (y_next - y_prev))[..., None]

    i0 = ig - 1
    i1 = ig
    j0 = jg - 1
    j1 = jg

    top_left = mapped[i0, j0]
    top_right = mapped[i0, j1]
    bottom_left = mapped[i1, j0]
    bottom_right = mapped[i1, j1]

    left = (1.0 - ty) * top_left + ty * bottom_left
    right = (1.0 - ty) * top_right + ty * bottom_right
    return (1.0 - tx) * left + tx * right


def _perimeter_samples(height: int, width: int, step: int = 16) -> np.ndarray:
    """Sample points along all edges of one image footprint."""

    top = np.stack(
        [np.arange(0, width, step), np.zeros(len(np.arange(0, width, step)))],
        axis=1,
    )
    bottom = np.stack(
        [
            np.arange(0, width, step),
            np.full(len(np.arange(0, width, step)), height - 1),
        ],
        axis=1,
    )
    left = np.stack(
        [
            np.zeros(len(np.arange(0, height, step))),
            np.arange(0, height, step),
        ],
        axis=1,
    )
    right = np.stack(
        [
            np.full(len(np.arange(0, height, step)), width - 1),
            np.arange(0, height, step),
        ],
        axis=1,
    )
    return np.unique(np.vstack([top, bottom, left, right]), axis=0).astype(
        np.float64
    )
