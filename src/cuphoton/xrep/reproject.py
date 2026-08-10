# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Public reprojection APIs for xRep."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .backends import get_backend, resolve_backend
from .geometry import (
    BBox,
    DeviceReprojectionResult,
    Grid,
    MaskedReprojectionResult,
    ReprojectionResult,
    ReprojectionSpec,
    StackReprojectionResult,
    StackReprojectionSpec,
    bbox_union,
)
from .io import load_fits_image_with_wcs, load_fits_mask
from .mapping import (
    PreparedReprojection,
    estimate_source_bbox_on_grid,
    make_grid_mapping,
    make_wcs_mapping,
    prepare_reprojection,
)


def _backend_unavailable_message(backend: str) -> str:
    extra = "gpu" if backend == "cupy" else "torch"
    return (
        f"Requested backend is unavailable: {backend}. Run "
        f"'uv sync --extra {extra}' for development or install "
        f"'cuphoton[{extra}]'."
    )


def _prepare_array_module(backend: str):
    """Pick the array module for prepare_reprojection's dense expansion.

    The cupy backend keeps the coordinate/area arrays on the GPU; all other
    backends prepare on the CPU with NumPy.
    """
    if backend == "cupy":
        try:
            import cupy as cp

            return cp
        except Exception:
            return np
    if backend == "torch":
        try:
            from .backends.torch_backend import make_torch_xp

            return make_torch_xp()
        except Exception:
            return np
    return np


def reproject_array(
    source: np.ndarray,
    spec: ReprojectionSpec,
    *,
    source_mask: np.ndarray | None = None,
    backend: str | None = None,
) -> ReprojectionResult:
    """Reproject one source array according to a reprojection specification.

    Parameters
    ----------
    source
        Two-dimensional source image.
    spec
        Destination mapping and interpolation settings.
    source_mask
        Optional source mask propagated to the output.
    backend
        ``cupy``, ``torch``, or ``cpu``. ``None`` selects the first available
        backend in that order.

    Returns
    -------
    ReprojectionResult
        Host-resident image, optional mask, and execution metadata.
    """

    backend = resolve_backend(backend)
    backend_impl = get_backend(backend)
    if not backend_impl.is_available():
        raise RuntimeError(_backend_unavailable_message(backend))
    prepared = prepare_reprojection(
        spec,
        source_shape=source.shape,
        xp=_prepare_array_module(backend),
    )
    return backend_impl.reproject(
        source,
        prepared,
        spec,
        source_mask=source_mask,
    )


def reproject_masked_array(
    image: np.ndarray,
    variance: np.ndarray,
    mask: np.ndarray,
    spec: ReprojectionSpec,
    *,
    backend: str | None = None,
    prepared: PreparedReprojection | None = None,
    variance_fill_value: float = float("nan"),
) -> MaskedReprojectionResult:
    """Reproject an image, diagonal variance, and bit-mask plane together.

    One prepared mapping is reused for every plane. Image interpolation uses
    the selected normalized kernel, variance uses the squares of those
    normalized weights, and mask bits use exact neighborhood bitwise OR.
    Relative-area scaling is applied once to image values and squared for
    variances.

    The variance result is a diagonal approximation: interpolation-induced
    covariance between destination pixels is not represented.

    Parameters
    ----------
    image, variance, mask
        Two-dimensional source planes with identical shapes. ``mask`` must
        have an integer or boolean dtype. Variance values must be
        nonnegative; NaN and positive infinity are accepted as no-data
        values.
    spec
        Destination mapping and interpolation settings.
    backend
        ``cupy``, ``torch``, or ``cpu``. ``None`` selects the first available
        backend in that order.
    prepared
        Optional reusable mapping prepared for ``spec`` and the source shape.
        As with :func:`reproject_array_device`, callers supplying a prepared
        mapping are responsible for using the source shape it was built for.
    variance_fill_value
        Variance value assigned outside the valid source footprint. This is
        independent of ``spec.fill_value`` because an image sentinel is not a
        valid variance estimate.

    Returns
    -------
    MaskedReprojectionResult
        Host-resident reprojected planes and execution metadata.
    """

    source_shape = tuple(int(value) for value in image.shape)
    if len(source_shape) != 2:
        raise ValueError("image must be a 2D array")
    if tuple(int(value) for value in variance.shape) != source_shape:
        raise ValueError("variance shape must match image shape")
    if tuple(int(value) for value in mask.shape) != source_shape:
        raise ValueError("mask shape must match image shape")
    variance_array = np.asarray(variance)
    if np.dtype(variance_array.dtype).kind not in {"f", "i", "u"}:
        raise TypeError("variance must have a real numeric dtype")
    if np.any(variance_array < 0.0):
        raise ValueError("variance must not contain negative values")
    if np.dtype(mask.dtype).kind not in {"b", "i", "u"}:
        raise TypeError("mask must have an integer or boolean dtype")

    backend = resolve_backend(backend)
    backend_impl = get_backend(backend)
    if not backend_impl.is_available():
        raise RuntimeError(_backend_unavailable_message(backend))
    if prepared is None:
        prepared = prepare_reprojection(
            spec,
            source_shape=source_shape,
            xp=_prepare_array_module(backend),
        )
    elif prepared.output_shape != spec.output_bbox.shape:
        raise ValueError("prepared output shape must match spec output bbox")

    image_result = backend_impl.reproject(
        image,
        prepared,
        spec,
        source_mask=None,
    )
    reprojected_variance = backend_impl.reproject_variance(
        variance_array,
        prepared,
        spec,
        variance_fill_value=variance_fill_value,
    )
    reprojected_mask = backend_impl.reproject_mask(mask, prepared, spec)

    metadata = dict(image_result.metadata)
    metadata.update(
        {
            "variance_propagation": "diagonal",
            "variance_covariance_propagated": False,
            "variance_area_scaling_power": (2 if spec.area_scaling else 0),
        }
    )
    return MaskedReprojectionResult(
        image=image_result.image,
        variance=np.asarray(reprojected_variance, dtype=np.float64),
        mask=np.asarray(reprojected_mask),
        bbox=image_result.bbox,
        backend=image_result.backend,
        interpolation=image_result.interpolation,
        metadata=metadata,
    )


def reproject_array_device(
    source,
    spec: ReprojectionSpec,
    *,
    source_mask=None,
    backend: str | None = "cupy",
    prepared: PreparedReprojection | None = None,
) -> DeviceReprojectionResult:
    """Reproject one array while keeping the result on the backend device.

    Parameters
    ----------
    source
        Backend-native two-dimensional source array.
    spec
        Destination mapping and interpolation settings.
    source_mask
        Optional backend-native source mask.
    backend
        Device backend. Only ``cupy`` is currently supported.
    prepared
        Reusable dense mapping for repeated reprojections with identical
        geometry.

    Returns
    -------
    DeviceReprojectionResult
        Device-resident image and optional mask.
    """

    backend = resolve_backend(backend)
    if backend != "cupy":
        raise NotImplementedError(
            "device-resident reprojections currently require the cupy backend"
        )
    backend_impl = get_backend(backend)
    if not backend_impl.is_available():
        raise RuntimeError(_backend_unavailable_message(backend))
    if prepared is None:
        prepared = prepare_reprojection(
            spec,
            source_shape=tuple(int(value) for value in source.shape),
            xp=_prepare_array_module(backend),
        )
    if not hasattr(backend_impl, "reproject_device"):
        raise RuntimeError(
            f"Backend does not support device reprojections: {backend}"
        )
    return backend_impl.reproject_device(
        source,
        prepared,
        spec,
        source_mask=source_mask,
    )


def reproject_fits(
    path: Path,
    *,
    grid: Grid | None = None,
    output_bbox: BBox | None = None,
    hdu: int | None = None,
    mask_path: Path | None = None,
    mask_hdu: int | None = None,
    interpolation: str = "lanczos3",
    backend: str | None = None,
    mapping_grid_step: int = 100,
    area_scaling: bool = True,
) -> ReprojectionResult:
    """Reproject one FITS image onto a shared sky grid.

    Parameters
    ----------
    path
        Source FITS image.
    grid
        Destination grid; derived from the source when omitted.
    output_bbox
        Optional destination footprint on ``grid``.
    hdu
        Explicit source image HDU.
    mask_path, mask_hdu
        Optional mask FITS path and HDU.
    interpolation
        ``lanczos3`` or ``bilinear``.
    backend
        ``cupy``, ``torch``, or ``cpu``. ``None`` auto-selects.
    mapping_grid_step
        Coarse WCS mapping interval in output pixels.
    area_scaling
        Apply relative pixel-area scaling.

    Returns
    -------
    ReprojectionResult
        Host-resident reprojected image, optional mask, and metadata.
    """

    image, source_wcs, _, _ = load_fits_image_with_wcs(path, hdu=hdu)
    if grid is None:
        grid = _default_grid_from_wcs(source_wcs, image.shape)

    mapping, bbox = make_wcs_mapping(
        path,
        grid,
        hdu=hdu,
        output_bbox=output_bbox,
    )

    mask = None
    if mask_path is not None:
        mask, _, _ = load_fits_mask(mask_path, hdu=mask_hdu)
    spec = ReprojectionSpec(
        mapping=mapping,
        output_bbox=bbox,
        interpolation=interpolation,
        mapping_grid_step=mapping_grid_step,
        area_scaling=area_scaling,
    )
    result = reproject_array(
        image,
        spec,
        source_mask=mask,
        backend=backend,
    )
    result.metadata.update(
        {
            "path": str(Path(path).expanduser().resolve()),
            "grid_crval": list(grid.crval),
            "grid_pixel_scale_arcsec": grid.pixel_scale_arcsec,
        }
    )
    return result


def reproject_stack(
    sources: list[np.ndarray] | tuple[np.ndarray, ...],
    spec: StackReprojectionSpec,
    *,
    source_masks: (
        list[np.ndarray | None] | tuple[np.ndarray | None, ...] | None
    ) = None,
    backend: str | None = None,
) -> StackReprojectionResult:
    """Reproject multiple source arrays onto one shared grid.

    Parameters
    ----------
    sources
        Source arrays in the same order as ``spec.members``.
    spec
        Shared-grid and per-source reprojection specifications.
    source_masks
        Optional masks in source order.
    backend
        ``cupy``, ``torch``, or ``cpu``. ``None`` auto-selects.

    Returns
    -------
    StackReprojectionResult
        Stacked host arrays and individual reprojection results.
    """

    backend = resolve_backend(backend)
    if len(sources) != len(spec.members):
        raise ValueError("source count must match the number of member specs")
    if source_masks is None:
        source_masks = [None] * len(sources)
    if len(source_masks) != len(sources):
        raise ValueError("source mask count must match source count")

    results = tuple(
        reproject_array(
            np.asarray(source),
            member,
            source_mask=mask,
            backend=backend,
        )
        for source, member, mask in zip(
            sources,
            spec.members,
            source_masks,
            strict=True,
        )
    )
    images = np.stack([result.image for result in results], axis=0)
    masks = None
    if any(result.mask is not None for result in results):
        masks = np.stack(
            [
                (
                    result.mask
                    if result.mask is not None
                    else np.zeros(spec.output_bbox.shape, dtype=bool)
                )
                for result in results
            ],
            axis=0,
        )
    return StackReprojectionResult(
        images=images,
        masks=masks,
        spec=spec,
        results=results,
        backend=backend,
        metadata={
            "grid_crval": list(spec.grid.crval),
            "grid_pixel_scale_arcsec": spec.grid.pixel_scale_arcsec,
        },
    )


def build_stack_spec_from_fits(
    paths: list[Path] | tuple[Path, ...],
    *,
    grid: Grid | None = None,
    hdu: int | None = None,
    interpolation: str = "lanczos3",
    mapping_grid_step: int = 100,
    area_scaling: bool = True,
) -> tuple[list[np.ndarray], StackReprojectionSpec]:
    """Load FITS images and build a common stack-reprojection specification.

    Parameters
    ----------
    paths
        Non-empty FITS paths in desired stack order.
    grid
        Destination grid; derived from the first image when omitted.
    hdu
        Explicit image HDU used for every input.
    interpolation
        Interpolation kernel for every member.
    mapping_grid_step
        Coarse WCS mapping interval in output pixels.
    area_scaling
        Apply relative pixel-area scaling.

    Returns
    -------
    images
        Loaded source images in path order.
    spec
        Stack specification with a union footprint and per-source mappings.
    """

    if not paths:
        raise ValueError(
            "build_stack_spec_from_fits requires at least one path"
        )
    payloads = [load_fits_image_with_wcs(path, hdu=hdu) for path in paths]
    if grid is None:
        grid = _default_grid_from_wcs(payloads[0][1], payloads[0][0].shape)

    bboxes = []
    for image, source_wcs, _, _ in payloads:
        bboxes.append(
            estimate_source_bbox_on_grid(
                source_wcs,
                shape=image.shape,
                grid=grid,
            )
        )
    union_bbox = bbox_union(bboxes)

    members = []
    for _, source_wcs, _, _ in payloads:
        mapping = make_grid_mapping(
            source_wcs,
            grid=grid,
            output_bbox=union_bbox,
        )
        members.append(
            ReprojectionSpec(
                mapping=mapping,
                output_bbox=union_bbox,
                interpolation=interpolation,
                mapping_grid_step=mapping_grid_step,
                area_scaling=area_scaling,
            )
        )
    return (
        [payload[0] for payload in payloads],
        StackReprojectionSpec(
            grid=grid,
            output_bbox=union_bbox,
            members=tuple(members),
        ),
    )


def _default_grid_from_wcs(source_wcs, shape: tuple[int, int]) -> Grid:
    center_x = (shape[1] - 1) / 2.0
    center_y = (shape[0] - 1) / 2.0
    if callable(getattr(source_wcs, "pixel_to_world", None)):
        center = source_wcs.pixel_to_world(center_x, center_y)
        longitude = center.spherical.lon.to_value("deg")
        latitude = center.spherical.lat.to_value("deg")
    else:
        # Preserve the original numeric-WCS compatibility contract. Celestial
        # low-level WCS values are assumed to be degrees, as before.
        center = np.asarray(
            source_wcs.all_pix2world([[center_x, center_y]], 0),
            dtype=np.float64,
        )[0]
        longitude = float(center[0])
        latitude = float(center[1])
    pixel_scale = float(
        np.sqrt(abs(np.linalg.det(source_wcs.pixel_scale_matrix))) * 3600.0
    )
    return Grid(
        crval=(float(longitude), float(latitude)),
        pixel_scale_arcsec=pixel_scale,
    )
