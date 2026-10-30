# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""xImage bundle metadata, loading, validation, and ROI utilities."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np


class ArrayProxy(Protocol):
    """Structural type for an on-disk HDF5 or Zarr array."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...

    @property
    def dtype(self) -> np.dtype[Any]: ...

    @property
    def nbytes(self) -> int: ...

    def __getitem__(self, key: Any) -> Any: ...


type XImageArray = np.ndarray | ArrayProxy


@dataclass(frozen=True)
class Artifact:
    """Describe one array in an xImage reconstruction bundle.

    Attributes
    ----------
    key
        Stable key used in bundle mappings and array-store datasets.
    axes
        Semantic axis names in storage order.
    spatial_dims
        Indices of the ``(y, x)`` axes, or ``None`` for non-spatial data.
    frame_dim
        Frame-axis index, or ``None`` when the artifact has no such axis.
    """

    key: str
    axes: tuple[str, ...]
    spatial_dims: tuple[int, int] | None
    frame_dim: int | None

    @property
    def npy_name(self) -> str:
        """Return the canonical NumPy filename for this artifact."""

        return f"{self.key}.npy"


ARTIFACTS: dict[str, Artifact] = {
    "images": Artifact(
        key="images",
        axes=("batch", "y", "x", "frame"),
        spatial_dims=(1, 2),
        frame_dim=3,
    ),
    "masks": Artifact(
        key="masks",
        axes=("batch", "y", "x", "frame"),
        spatial_dims=(1, 2),
        frame_dim=3,
    ),
    "sky": Artifact(
        key="sky",
        axes=("batch", "y", "x", "frame"),
        spatial_dims=(1, 2),
        frame_dim=3,
    ),
    "variances": Artifact(
        key="variances",
        axes=("batch", "y", "x", "frame"),
        spatial_dims=(1, 2),
        frame_dim=3,
    ),
    "psfs": Artifact(
        key="psfs",
        axes=("psf_y", "psf_x", "channel", "frame"),
        spatial_dims=(0, 1),
        frame_dim=3,
    ),
    "mcgauss": Artifact(
        key="mcgauss",
        axes=("frame", "kernel_y", "kernel_x"),
        spatial_dims=(1, 2),
        frame_dim=0,
    ),
    "exp_times": Artifact(
        key="exp_times",
        axes=("frame",),
        spatial_dims=None,
        frame_dim=0,
    ),
}


_REQUIRED_ARTIFACT_KEYS = ("images", "variances", "psfs")


_RECONSTRUCTION_ARTIFACT_KEYS = (
    "images",
    "masks",
    "sky",
    "variances",
    "psfs",
    "exp_times",
    "mcgauss",
)


STORE_PRIORITY = ("npy", "hdf5", "zarr")


@dataclass
class XImageBundle(Mapping[str, XImageArray]):
    """Collection of xImage arrays loaded from one supported store.

    Images, variances, and PSFs are required. An optional boolean ``masks``
    array matches the image shape and uses ``True`` for valid pixels.
    Missing masks treat all pixels as valid; missing ``exp_times`` use one
    per frame. ``sky`` and ``mcgauss`` are optional auxiliary arrays.

    HDF5 and Zarr arrays remain lazy proxies. Call :meth:`close` when the
    bundle is no longer needed, or use it as a context manager.

    Attributes
    ----------
    store_format
        Detected storage format: ``npy``, ``hdf5``, or ``zarr``.
    store_path
        Path to the directory or container file backing the bundle.
    arrays
        Loaded NumPy arrays or lazy array proxies keyed by artifact name.
    metadata
        Store metadata loaded from the adjacent JSON document when present.
    resources
        Backend-owned objects retained for the lifetime of lazy proxies.
    """

    store_format: str
    store_path: Path
    arrays: dict[str, XImageArray]
    metadata: dict[str, Any]
    resources: tuple[object, ...] = ()

    def __getitem__(self, key: str) -> XImageArray:
        return self.arrays[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.arrays)

    def __len__(self) -> int:
        return len(self.arrays)

    @property
    def images(self) -> XImageArray:
        return self.arrays["images"]

    @property
    def masks(self) -> XImageArray:
        if "masks" in self.arrays:
            return self.arrays["masks"]
        return np.broadcast_to(np.array(True), self.images.shape)

    @property
    def sky(self) -> XImageArray | None:
        return self.arrays.get("sky")

    @property
    def variances(self) -> XImageArray:
        return self.arrays["variances"]

    @property
    def psfs(self) -> XImageArray:
        return self.arrays["psfs"]

    @property
    def mcgauss(self) -> XImageArray | None:
        return self.arrays.get("mcgauss")

    @property
    def exp_times(self) -> XImageArray:
        if "exp_times" in self.arrays:
            return self.arrays["exp_times"]
        return np.ones(self.images.shape[3], dtype=np.float64)

    def axes(self, key: str) -> tuple[str, ...]:
        """Return semantic axis names for one loaded artifact."""

        return ARTIFACTS[key].axes

    def roi(
        self,
        key: str,
        *,
        center: tuple[int, int] | None = None,
        height: int | None = None,
        width: int | None = None,
        copy: bool = False,
    ) -> np.ndarray:
        """Return a centered spatial view or copy of one artifact."""

        return slice_spatial_roi(
            self.arrays[key],
            ARTIFACTS[key],
            center=center,
            height=height,
            width=width,
            copy=copy,
        )

    def reconstruction_inputs(self) -> tuple[XImageArray | None, ...]:
        """Return solver arrays, using None for absent auxiliary data."""

        return (
            self.images,
            self.psfs,
            self.masks,
            self.sky,
            self.variances,
            self.exp_times,
            self.mcgauss,
        )

    def select_exposures(
        self, indices: list[int] | tuple[int, ...]
    ) -> XImageBundle:
        """Return a bundle restricted to selected exposure indices."""

        arrays: dict[str, XImageArray] = {}
        for key, arr in self.arrays.items():
            artifact = ARTIFACTS[key]
            if artifact.frame_dim is None:
                arrays[key] = arr
                continue
            slicer: list[slice | list[int]] = [slice(None)] * arr.ndim
            slicer[artifact.frame_dim] = list(indices)
            arrays[key] = arr[tuple(slicer)]
        metadata = dict(self.metadata)
        metadata["selected_exposures"] = list(indices)
        return XImageBundle(
            store_format=self.store_format,
            store_path=self.store_path,
            arrays=arrays,
            metadata=metadata,
            resources=self.resources,
        )

    def describe(self) -> dict[str, dict[str, Any]]:
        """Return shape, dtype, axes, and byte size for loaded artifacts."""

        return summarize_bundle(self)

    def close(self) -> None:
        """Close any backend resources retained by lazy array proxies."""

        for resource in self.resources:
            close = getattr(resource, "close", None)
            if callable(close):
                close()

    def __enter__(self) -> XImageBundle:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def resolve_artifact_keys(
    spec: str | list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Normalize and validate an artifact selection.

    Parameters
    ----------
    spec
        ``None`` or ``"all"`` for every artifact, a comma-separated string,
        or an explicit sequence of keys.

    Returns
    -------
    list of str
        Artifact keys in requested order.
    """

    if spec is None or spec == "all":
        return list(ARTIFACTS)
    if isinstance(spec, str):
        keys = [key.strip() for key in spec.split(",") if key.strip()]
    else:
        keys = list(spec)
    unknown = [key for key in keys if key not in ARTIFACTS]
    if unknown:
        raise ValueError(f"Unknown artifact keys: {', '.join(unknown)}")
    return keys


def expected_rank(artifact: Artifact) -> int:
    return len(artifact.axes)


def detect_ximage_store(
    base: str | Path = "data",
    *,
    prefer: tuple[str, ...] = STORE_PRIORITY,
) -> tuple[str, Path]:
    """Identify the format of an xImage array store.

    Parameters
    ----------
    base
        Path to an NPY directory, HDF5 file, or Zarr store.
    prefer
        Storage formats in descending preference order.

    Returns
    -------
    tuple of str and pathlib.Path
        Detected format name and resolved candidate path.

    Raises
    ------
    FileNotFoundError
        If no supported store can be found.
    """

    base_path = Path(base)
    candidates: dict[str, Path] = {}

    if looks_like_hdf5_store(base_path):
        candidates["hdf5"] = base_path
    if looks_like_zarr_store(base_path):
        candidates["zarr"] = base_path
    if looks_like_npy_store(base_path):
        candidates["npy"] = base_path

    for fmt in prefer:
        if fmt in candidates:
            return fmt, candidates[fmt]

    raise FileNotFoundError(
        f"Could not find an xImage store at {base_path}. "
        "Expected an NPY directory, HDF5 file, or Zarr store."
    )


def looks_like_hdf5_store(path: Path) -> bool:
    return path.is_file() and path.suffix in {".h5", ".hdf5"}


def looks_like_zarr_store(path: Path) -> bool:
    return path.is_dir() and (path / "zarr.json").exists()


def looks_like_npy_store(path: Path) -> bool:
    return path.is_dir() and any(
        (path / artifact.npy_name).exists() for artifact in ARTIFACTS.values()
    )


def load_ximage_bundle(
    base: str | Path = "data",
    *,
    prefer: tuple[str, ...] = STORE_PRIORITY,
    mmap_mode: Literal["r+", "r", "w+", "c"] | None = "r",
    artifacts: str | list[str] | tuple[str, ...] | None = None,
) -> XImageBundle:
    """Load selected arrays from an xImage store.

    Parameters
    ----------
    base
        Path to an NPY directory, HDF5 file, or Zarr store.
    prefer
        Storage formats in descending preference order.
    mmap_mode
        NumPy memory-map mode for ``.npy`` arrays, or ``None`` to load them.
    artifacts
        Artifact keys to load. ``None`` or ``"all"`` loads images, variances,
        and PSFs plus any available optional arrays in solver-input order.
        An explicit selection requires each named array to exist.

    Returns
    -------
    XImageBundle
        Bundle containing NumPy arrays or lazy HDF5/Zarr proxies.
    """

    store_format, store_path = detect_ximage_store(base, prefer=prefer)
    load_available = artifacts is None or artifacts == "all"
    keys = resolve_artifact_keys(
        _RECONSTRUCTION_ARTIFACT_KEYS if load_available else artifacts
    )
    resources: list[Any] = []
    try:
        if store_format == "hdf5":
            import h5py

            handle = h5py.File(store_path, "r")
            resources.append(handle)
            arrays = {
                key: handle[key]
                for key in keys
                if not load_available
                or key in _REQUIRED_ARTIFACT_KEYS
                or key in handle
            }
        elif store_format == "zarr":
            import zarr

            group = zarr.open_group(str(store_path), mode="r")
            resources.append(group)
            arrays = {
                key: group[key]
                for key in keys
                if not load_available
                or key in _REQUIRED_ARTIFACT_KEYS
                or key in group
            }
        else:
            arrays = {
                key: load_artifact(
                    store_path,
                    ARTIFACTS[key],
                    store_format=store_format,
                    mmap_mode=mmap_mode,
                )
                for key in keys
                if not load_available
                or key in _REQUIRED_ARTIFACT_KEYS
                or (store_path / ARTIFACTS[key].npy_name).exists()
            }
        metadata = load_store_metadata(store_format, store_path)
    except Exception:
        for resource in reversed(resources):
            close = getattr(resource, "close", None)
            if callable(close):
                close()
        raise
    return XImageBundle(
        store_format=store_format,
        store_path=store_path,
        arrays=arrays,
        metadata=metadata,
        resources=tuple(resources),
    )


def load_reconstruction_inputs(
    base: str | Path = "data",
    *,
    prefer: tuple[str, ...] = STORE_PRIORITY,
    mmap_mode: Literal["r+", "r", "w+", "c"] | None = "r",
) -> tuple[np.ndarray | None, ...]:
    """Load reconstruction arrays in solver argument order.

    Parameters
    ----------
    base
        Path to an NPY directory, HDF5 file, or Zarr store.
    prefer
        Storage formats in descending preference order.
    mmap_mode
        NumPy memory-map mode for ``.npy`` arrays.

    Returns
    -------
    tuple
        Images, PSFs, masks, sky, variances, exposure times, and Gaussian
        mixture kernels. Missing sky and Gaussian kernels remain ``None``;
        masks and exposure times use their bundle defaults.

    Notes
    -----
    HDF5 and Zarr proxies are materialized before their owning bundle is
    closed. Use :func:`load_ximage_bundle` as a context manager when lazy
    access is preferable.
    """

    bundle = load_ximage_bundle(base, prefer=prefer, mmap_mode=mmap_mode)
    try:
        return tuple(
            np.asarray(array) if array is not None else None
            for array in bundle.reconstruction_inputs()
        )
    finally:
        bundle.close()


def load_artifact(
    store_path: Path,
    artifact: Artifact,
    *,
    store_format: str,
    mmap_mode: Literal["r+", "r", "w+", "c"] | None = "r",
) -> np.ndarray:
    if store_format == "npy":
        return np.load(
            store_path / artifact.npy_name,
            mmap_mode=mmap_mode,
            allow_pickle=False,
        )
    if store_format in {"hdf5", "zarr"}:
        raise ValueError(
            f"Use load_ximage_bundle() for {store_format} stores."
        )
    raise ValueError(f"Unsupported store format: {store_format}")


def load_store_metadata(
    store_format: str, store_path: Path
) -> dict[str, Any]:
    if store_format == "npy":
        metadata_path = store_path / "metadata.json"
        if metadata_path.exists():
            return json.loads(metadata_path.read_text())
    if store_format == "zarr":
        metadata_path = store_path / "metadata.json"
        if metadata_path.exists():
            return json.loads(metadata_path.read_text())
    if store_format == "hdf5":
        metadata_path = store_path.with_suffix(store_path.suffix + ".json")
        if metadata_path.exists():
            return json.loads(metadata_path.read_text())
    return {
        "format": store_format,
        "source_dir": str(store_path.resolve()),
        "artifacts": {
            key: {
                "shape": None,
                "dtype": None,
                "axes": list(artifact.axes),
            }
            for key, artifact in ARTIFACTS.items()
        },
    }


def validate_bundle(bundle: XImageBundle) -> list[str]:
    """Check array dimensions, PSF geometry, and positive exposure times.

    Parameters
    ----------
    bundle
        Bundle to validate.

    Returns
    -------
    list of str
        Human-readable validation issues; an empty list means validation
        succeeded.
    """

    issues: list[str] = []
    missing = [key for key in _REQUIRED_ARTIFACT_KEYS if key not in bundle]
    if missing:
        issues.append(f"Missing required artifacts: {', '.join(missing)}")
        return issues

    for key, artifact in ARTIFACTS.items():
        if key not in bundle:
            continue
        arr = bundle[key]
        rank = expected_rank(artifact)
        if arr.ndim != rank:
            issues.append(f"{key} must be rank-{rank}, saw shape {arr.shape}")
            return issues

    images = bundle.images
    psfs = bundle.psfs
    if (
        psfs.shape[0] <= 0
        or psfs.shape[0] != psfs.shape[1]
        or psfs.shape[0] % 2 == 0
    ):
        issues.append(
            "PSF kernels must have positive, square, odd spatial dimensions, "
            f"saw {psfs.shape[:2]}"
        )
    exposure_times = np.asarray(bundle.exp_times)
    if exposure_times.dtype.kind not in "fiu" or not np.all(
        np.isfinite(exposure_times) & (exposure_times > 0)
    ):
        issues.append("Exposure times must be finite and strictly positive")

    for key in ("variances", "sky", "masks"):
        if key in bundle and bundle[key].shape != images.shape:
            issues.append(
                f"{key} shape {bundle[key].shape} does not match "
                f"images shape {images.shape}"
            )
    if "masks" in bundle and bundle.masks.dtype != np.dtype(bool):
        issues.append(
            "Masks must have boolean dtype with True for valid pixels"
        )

    frames = images.shape[3]
    for key in ("psfs", "mcgauss", "exp_times"):
        if key not in bundle:
            continue
        frame_dim = ARTIFACTS[key].frame_dim
        assert frame_dim is not None
        count = bundle[key].shape[frame_dim]
        if count != frames:
            issues.append(
                f"{key} frame count {count} != image frame count {frames}"
            )
    return issues


def summarize_bundle(bundle: XImageBundle) -> dict[str, dict[str, Any]]:
    """Summarize the storage contract of every loaded artifact.

    Parameters
    ----------
    bundle
        Bundle to describe.

    Returns
    -------
    dict
        Per-artifact shape, dtype, semantic axes, and byte size.
    """

    return {
        key: {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "axes": list(ARTIFACTS[key].axes),
            "nbytes": int(arr.nbytes),
        }
        for key, arr in bundle.arrays.items()
    }


def spatial_roi_slices(
    shape: tuple[int, ...],
    artifact: Artifact,
    *,
    center: tuple[int, int] | None = None,
    height: int | None = None,
    width: int | None = None,
) -> tuple[slice | int, ...]:
    """Build centered spatial slices for an artifact shape.

    Parameters
    ----------
    shape
        Full artifact shape.
    artifact
        Artifact metadata identifying spatial axes.
    center
        Optional ``(y, x)`` center in pixels; defaults to the array center.
    height, width
        Requested ROI size; omitted dimensions span the available extent.

    Returns
    -------
    tuple of slice
        Index tuple preserving all non-spatial axes.
    """

    if artifact.spatial_dims is None:
        raise ValueError(f"{artifact.key} has no spatial dimensions.")

    y_dim, x_dim = artifact.spatial_dims
    y_size = shape[y_dim]
    x_size = shape[x_dim]
    if center is None:
        center = (y_size // 2, x_size // 2)
    if height is None:
        height = y_size
    if width is None:
        width = x_size
    if height <= 0 or width <= 0:
        raise ValueError("ROI height and width must be positive.")
    if height > y_size or width > x_size:
        raise ValueError(
            f"Requested ROI {(height, width)} exceeds available spatial "
            f"shape {(y_size, x_size)}."
        )

    y_start = center[0] - height // 2
    x_start = center[1] - width // 2
    y_stop = y_start + height
    x_stop = x_start + width
    if y_start < 0 or x_start < 0 or y_stop > y_size or x_stop > x_size:
        raise ValueError(
            f"ROI center={center}, size={(height, width)} is out of bounds "
            f"for {(y_size, x_size)}."
        )

    slices: list[slice | int] = [slice(None)] * len(shape)
    slices[y_dim] = slice(y_start, y_stop)
    slices[x_dim] = slice(x_start, x_stop)
    return tuple(slices)


def slice_spatial_roi(
    array: XImageArray,
    artifact: Artifact,
    *,
    center: tuple[int, int] | None = None,
    height: int | None = None,
    width: int | None = None,
    copy: bool = False,
) -> np.ndarray:
    """Extract a centered spatial ROI from an artifact array.

    Parameters
    ----------
    array
        NumPy array or lazy array proxy.
    artifact
        Artifact metadata identifying spatial axes.
    center
        Optional ``(y, x)`` center in pixels.
    height, width
        Requested ROI dimensions.
    copy
        Copy the selected data instead of returning a view when possible.

    Returns
    -------
    numpy.ndarray
        Selected spatial region with non-spatial axes preserved.
    """

    slices = spatial_roi_slices(
        array.shape,
        artifact,
        center=center,
        height=height,
        width=width,
    )
    view = array[slices]
    if copy:
        return np.array(view, copy=True)
    return view


def centered_bundle_rois(
    bundle: XImageBundle,
    *,
    keys: tuple[str, ...] | None = None,
    center: tuple[int, int] | None = None,
    height: int | None = None,
    width: int | None = None,
    copy: bool = False,
) -> dict[str, np.ndarray]:
    """Extract the same centered spatial ROI from several artifacts.

    Parameters
    ----------
    bundle
        Source bundle.
    keys
        Spatial artifact keys to extract. By default, extract whichever of
        images, masks, sky, and variances are present in the store.
    center
        Optional ``(y, x)`` center shared by every artifact.
    height, width
        Requested ROI dimensions.
    copy
        Copy selected data instead of returning views when possible.

    Returns
    -------
    dict
        NumPy arrays keyed by artifact name.
    """

    if keys is None:
        keys = tuple(
            key
            for key in ("images", "masks", "sky", "variances")
            if key in bundle
        )
    missing = [key for key in keys if key not in bundle]
    if missing:
        loaded = ", ".join(bundle.keys())
        raise ValueError(
            f"Requested ROI keys not loaded in bundle: {', '.join(missing)}. "
            f"Loaded keys: {loaded}"
        )
    return {
        key: bundle.roi(
            key,
            center=center,
            height=height,
            width=width,
            copy=copy,
        )
        for key in keys
    }


def normalize_exposure_indices(
    indices: list[int] | tuple[int, ...] | None,
    *,
    max_exposures: int | None,
    total_exposures: int,
) -> list[int] | None:
    if indices is not None and max_exposures is not None:
        raise ValueError(
            "Specify either exposure_indices or max_exposures, not both."
        )
    if indices is not None:
        normalized = list(indices)
    elif max_exposures is not None:
        normalized = list(range(min(max_exposures, total_exposures)))
    else:
        return None
    if not normalized:
        raise ValueError("Exposure selection must not be empty.")
    for idx in normalized:
        if idx < 0 or idx >= total_exposures:
            raise ValueError(
                f"Exposure index {idx} is out of bounds for "
                f"{total_exposures} exposures."
            )
    return normalized
