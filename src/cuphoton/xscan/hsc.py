# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Local HSC NPY-store helpers used by XScan."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SPLIT_TO_INDEX = {
    "train": 0,
    "val": 1,
    "test": 2,
}
INDEX_TO_SPLIT = {value: key for key, value in SPLIT_TO_INDEX.items()}


@dataclass(slots=True)
class HscNpyStore:
    root: Path
    images: np.ndarray
    variances: np.ndarray
    masks: np.ndarray | None
    sky: np.ndarray | None
    psfs: np.ndarray
    exp_times: np.ndarray
    metadata: dict[str, Any]

    @classmethod
    def open(cls, base: str | Path) -> "HscNpyStore":
        root = resolve_hsc_npy_dir(base)
        metadata = json.loads((root / "metadata.json").read_text())
        images = np.load(root / "images.npy", mmap_mode="r")
        variances = np.load(root / "variances.npy", mmap_mode="r")
        masks = (
            np.load(root / "masks.npy", mmap_mode="r")
            if (root / "masks.npy").exists()
            else None
        )
        sky = (
            np.load(root / "sky.npy", mmap_mode="r")
            if (root / "sky.npy").exists()
            else None
        )
        psfs = np.load(root / "psfs.npy", mmap_mode="r")
        exp_times = np.load(root / "exp_times.npy", mmap_mode="r")
        return cls(
            root=root,
            images=images,
            variances=variances,
            masks=masks,
            sky=sky,
            psfs=psfs,
            exp_times=exp_times,
            metadata=metadata,
        )

    @property
    def spatial_shape(self) -> tuple[int, int]:
        return int(self.images.shape[1]), int(self.images.shape[2])

    @property
    def exposure_count(self) -> int:
        return int(self.images.shape[-1])


@dataclass(slots=True)
class HscCatalogCandidate:
    object_id: int | None
    center_x: int
    center_y: int
    original_x: float
    original_y: float
    tract: int | None
    patch: str | None
    ref_band: str | None
    is_primary: bool | None
    is_sky: bool | None
    flux: float | None
    flux_column: str | None
    extendedness: float | None


@dataclass(slots=True)
class HscXPOISConfig:
    kernel_shape: tuple[int, int] = (9, 9)
    basis_sigmas: tuple[float, ...] = (1.5, 3.0)
    basis_degrees: tuple[int, ...] = (2, 1)
    background_degree: int = 0
    flux_conserve: bool = False
    use_variance: bool = True
    solver_mode: str = "constant"


HSC_MASK_PLANE_INDEX = {
    "chip_gap_border_1": 0,
    "stripes_1": 1,
    "chip_gap_border_2": 2,
    "small_sources": 3,
    "chip_gap_border_3": 4,
    "various_sources": 5,
    "null_1": 6,
    "stripes_2": 7,
    "chip_gap": 8,
    "stripes_3": 9,
    "large_sources": 10,
    "null_2": 11,
}


def resolve_hsc_npy_dir(base: str | Path) -> Path:
    base_path = Path(base).expanduser().resolve()
    candidates = [
        base_path,
        base_path / "HSC_npy",
    ]
    for candidate in candidates:
        if looks_like_hsc_npy_dir(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find an HSC_npy directory under "
        f"{base_path}. Tried: {', '.join(str(item) for item in candidates)}"
    )


def _import_xpois_symbols():
    from cuphoton.xpois import (
        GaussianBasisComponent,
        solve_constant_kernel,
        solve_separable_kernel,
    )

    return (
        GaussianBasisComponent,
        solve_constant_kernel,
        solve_separable_kernel,
    )


def looks_like_hsc_npy_dir(path: Path) -> bool:
    return path.is_dir() and all(
        (path / name).exists()
        for name in (
            "images.npy",
            "variances.npy",
            "psfs.npy",
            "exp_times.npy",
            "metadata.json",
        )
    )


def resolve_hsc_catalog_parquet(
    base: str | Path,
    *,
    explicit_path: str | Path | None = None,
) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"HSC catalog parquet not found: {path}")
        return path

    npy_root = resolve_hsc_npy_dir(base)
    data_root = npy_root.parent
    candidates = sorted(
        (data_root / "HSC" / "FITS" / "catalogs").rglob("*.parq")
    )
    candidates.extend(
        sorted((data_root / "HSC" / "FITS" / "catalogs").rglob("*.parquet"))
    )
    if not candidates:
        raise FileNotFoundError(
            "Could not find an HSC object-table parquet under "
            f"{data_root / 'HSC' / 'FITS' / 'catalogs'}"
        )
    if len(candidates) > 1:
        raise FileExistsError(
            "Multiple HSC catalog parquet files found; pass catalog_parquet "
            "explicitly in the manifest."
        )
    return candidates[0]


def resolve_hsc_coadd_fits(
    base: str | Path,
    *,
    explicit_path: str | Path | None = None,
) -> Path | None:
    if explicit_path is not None:
        path = Path(explicit_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"HSC coadd FITS not found: {path}")
        return path

    npy_root = resolve_hsc_npy_dir(base)
    data_root = npy_root.parent
    candidates = sorted(
        (data_root / "HSC" / "FITS" / "coadd").rglob("*.fits")
    )
    if not candidates:
        return None
    if len(candidates) > 1:
        raise FileExistsError(
            "Multiple HSC coadd FITS files found; pass coadd_fits explicitly "
            "in the manifest."
        )
    return candidates[0]


def load_hsc_patch_origin(
    base: str | Path,
    *,
    coadd_fits: str | Path | None = None,
) -> tuple[float, float]:
    path = resolve_hsc_coadd_fits(base, explicit_path=coadd_fits)
    if path is None:
        return 0.0, 0.0

    try:
        from astropy.io import fits
    except ImportError as exc:
        raise ValueError(
            "astropy is required to read HSC coadd FITS metadata"
        ) from exc

    with fits.open(path, memmap=True) as hdul:
        header = hdul[1].header
        if "CRVAL1A" in header and "CRVAL2A" in header:
            return float(header["CRVAL1A"]), float(header["CRVAL2A"])
        if "LTV1" in header and "LTV2" in header:
            return float(-header["LTV1"]), float(-header["LTV2"])
    return 0.0, 0.0


def default_hsc_flux_column(band: str) -> str:
    clean = band.strip().lower()
    if not clean:
        raise ValueError("band must not be empty")
    return f"{clean}_psfFlux"


def default_hsc_extendedness_column(band: str) -> str:
    clean = band.strip().lower()
    if not clean:
        raise ValueError("band must not be empty")
    return f"{clean}_extendedness"


def load_hsc_catalog_candidates(
    base: str | Path,
    *,
    catalog_parquet: str | Path | None = None,
    coadd_fits: str | Path | None = None,
    band: str = "i",
    flux_column: str | None = None,
    primary_only: bool = True,
    exclude_sky: bool = True,
    flux_min: float | None = None,
    flux_max: float | None = None,
    extendedness_min: float | None = None,
    extendedness_max: float | None = None,
    limit: int | None = None,
    stamp_size: int | None = None,
) -> tuple[list[HscCatalogCandidate], Path]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError(
            "pyarrow is required for HSC catalog-aware dataset building"
        ) from exc

    catalog_path = resolve_hsc_catalog_parquet(
        base,
        explicit_path=catalog_parquet,
    )
    origin_x, origin_y = load_hsc_patch_origin(
        base,
        coadd_fits=coadd_fits,
    )
    resolved_flux_column = flux_column or default_hsc_flux_column(band)
    extendedness_column = default_hsc_extendedness_column(band)
    table = pq.read_table(catalog_path)
    payload = table.to_pydict()
    names = set(payload)

    required = {"x", "y"}
    missing = sorted(required - names)
    if missing:
        raise ValueError(
            "HSC catalog parquet is missing required columns: "
            + ", ".join(missing)
        )

    half = stamp_size // 2 if stamp_size is not None else None
    candidates: list[HscCatalogCandidate] = []
    x_values = payload["x"]
    y_values = payload["y"]
    row_count = len(x_values)
    for idx in range(row_count):
        raw_x = x_values[idx]
        raw_y = y_values[idx]
        if raw_x is None or raw_y is None:
            continue
        try:
            original_x = float(raw_x)
            original_y = float(raw_y)
        except Exception:
            continue
        if not np.isfinite(original_x) or not np.isfinite(original_y):
            continue
        center_x = int(round(original_x - origin_x))
        center_y = int(round(original_y - origin_y))
        if half is not None and (center_x - half < 0 or center_y - half < 0):
            continue
        is_primary = (
            bool(payload["detect_isPrimary"][idx])
            if "detect_isPrimary" in names
            and payload["detect_isPrimary"][idx] is not None
            else None
        )
        if primary_only and is_primary is False:
            continue

        sky_object = (
            bool(payload["sky_object"][idx])
            if "sky_object" in names
            and payload["sky_object"][idx] is not None
            else False
        )
        merge_peak_sky = (
            bool(payload["merge_peak_sky"][idx])
            if "merge_peak_sky" in names
            and payload["merge_peak_sky"][idx] is not None
            else False
        )
        is_sky = bool(sky_object or merge_peak_sky)
        if exclude_sky and is_sky:
            continue

        flux = None
        if resolved_flux_column in names:
            raw_flux = payload[resolved_flux_column][idx]
            if raw_flux is not None:
                try:
                    flux = float(raw_flux)
                except Exception:
                    flux = None
        if flux_min is not None and (flux is None or not np.isfinite(flux)):
            continue
        if flux_min is not None and flux < flux_min:
            continue
        if flux_max is not None and (flux is None or not np.isfinite(flux)):
            continue
        if flux_max is not None and flux > flux_max:
            continue

        extendedness = None
        if extendedness_column in names:
            raw_extendedness = payload[extendedness_column][idx]
            if raw_extendedness is not None:
                try:
                    extendedness = float(raw_extendedness)
                except Exception:
                    extendedness = None
        if extendedness_min is not None and (
            extendedness is None or not np.isfinite(extendedness)
        ):
            continue
        if extendedness_min is not None and extendedness < extendedness_min:
            continue
        if extendedness_max is not None and (
            extendedness is None or not np.isfinite(extendedness)
        ):
            continue
        if extendedness_max is not None and extendedness > extendedness_max:
            continue

        object_id = None
        if "objectId" in names and payload["objectId"][idx] is not None:
            try:
                object_id = int(payload["objectId"][idx])
            except Exception:
                object_id = None

        tract = None
        if "tract" in names and payload["tract"][idx] is not None:
            try:
                tract = int(payload["tract"][idx])
            except Exception:
                tract = None

        patch = None
        if "patch" in names and payload["patch"][idx] is not None:
            patch = str(payload["patch"][idx])

        ref_band = None
        if "refBand" in names and payload["refBand"][idx] is not None:
            ref_band = str(payload["refBand"][idx])

        candidates.append(
            HscCatalogCandidate(
                object_id=object_id,
                center_x=center_x,
                center_y=center_y,
                original_x=original_x,
                original_y=original_y,
                tract=tract,
                patch=patch,
                ref_band=ref_band,
                is_primary=is_primary,
                is_sky=is_sky,
                flux=flux,
                flux_column=(
                    resolved_flux_column
                    if resolved_flux_column in names
                    else None
                ),
                extendedness=extendedness,
            )
        )

    if limit is not None:
        if limit <= 0:
            raise ValueError("catalog_limit must be positive when provided")
        candidates = candidates[:limit]
    if not candidates:
        raise ValueError(
            "No HSC catalog candidates remained after applying the requested "
            "filters"
        )
    return candidates, catalog_path


def create_hsc_valid_masks(
    hsc_masks: np.ndarray | None,
    *,
    mode: str = "none",
    dilation_factor: int = 1,
) -> np.ndarray | None:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"", "none", "off", "false"}:
        return None
    if hsc_masks is None:
        raise ValueError(
            "HSC mask mode was requested, but masks.npy is not available"
        )
    if dilation_factor <= 0:
        raise ValueError("mask_dilation_factor must be positive")

    masks = np.asarray(hsc_masks, dtype=bool)
    if masks.ndim != 5 or masks.shape[0] != 1 or masks.shape[3] < 12:
        raise ValueError(
            "HSC masks must have shape "
            "(1, height, width, mask_class, exposure)"
        )

    def plane(name: str) -> np.ndarray:
        return masks[:, :, :, HSC_MASK_PLANE_INDEX[name], :]

    excluded = (
        plane("chip_gap_border_1")
        | plane("chip_gap_border_3")
        | plane("chip_gap")
    )
    if normalized == "non_conservative":
        return np.logical_not(excluded)
    if normalized != "conservative":
        raise ValueError(
            "mask_mode must be one of: none, non_conservative, conservative"
        )

    large_sources = np.array(plane("large_sources"), copy=True)
    if dilation_factor > 1:
        from scipy.ndimage import binary_dilation

        structure = np.ones((dilation_factor, dilation_factor), dtype=bool)
        for exposure_index in range(large_sources.shape[-1]):
            large_sources[0, :, :, exposure_index] = binary_dilation(
                large_sources[0, :, :, exposure_index],
                structure=structure,
                iterations=1,
            )
    excluded = excluded | large_sources
    for name in ("stripes_1", "stripes_2", "stripes_3"):
        excluded = excluded | plane(name)
    return np.logical_not(excluded)


def extract_stamp(
    image: np.ndarray, center_y: int, center_x: int, stamp_size: int
) -> np.ndarray:
    half = stamp_size // 2
    y0 = center_y - half
    y1 = center_y + half + 1
    x0 = center_x - half
    x1 = center_x + half + 1
    if y0 < 0 or x0 < 0 or y1 > image.shape[0] or x1 > image.shape[1]:
        raise ValueError("stamp exceeds image bounds")
    return np.asarray(image[y0:y1, x0:x1], dtype=np.float32)


def template_from_stack(
    images: np.ndarray,
    center_y: int,
    center_x: int,
    stamp_size: int,
    *,
    exclude_exposure: int | None = None,
    valid_masks: np.ndarray | None = None,
    min_valid_exposures: int = 1,
) -> np.ndarray:
    if min_valid_exposures <= 0:
        raise ValueError("min_valid_exposures must be positive")
    half = stamp_size // 2
    stack = np.asarray(
        images[
            0,
            center_y - half : center_y + half + 1,
            center_x - half : center_x + half + 1,
            :,
        ],
        dtype=np.float32,
    )
    mask_stack = None
    if valid_masks is not None:
        mask_stack = np.asarray(
            valid_masks[
                0,
                center_y - half : center_y + half + 1,
                center_x - half : center_x + half + 1,
                :,
            ],
            dtype=bool,
        )
    if exclude_exposure is not None and stack.shape[-1] > 1:
        keep = [
            idx for idx in range(stack.shape[-1]) if idx != exclude_exposure
        ]
        stack = stack[:, :, keep]
        if mask_stack is not None:
            mask_stack = mask_stack[:, :, keep]
    if mask_stack is None:
        return np.median(stack, axis=-1).astype(np.float32)
    valid_counts = np.sum(mask_stack, axis=-1)
    masked_stack = np.where(mask_stack, stack, np.nan)
    with np.errstate(all="ignore"):
        template = np.nanmedian(masked_stack, axis=-1)
    template[valid_counts < min_valid_exposures] = np.nan
    return template.astype(np.float32)


def extract_valid_mask_stamp(
    valid_masks: np.ndarray | None,
    center_y: int,
    center_x: int,
    stamp_size: int,
    exposure_index: int,
) -> np.ndarray | None:
    if valid_masks is None:
        return None
    return extract_stamp(
        valid_masks[0, :, :, exposure_index],
        center_y,
        center_x,
        stamp_size,
    ).astype(bool)


def valid_fraction(mask: np.ndarray | None) -> float | None:
    if mask is None:
        return None
    return float(np.mean(np.asarray(mask, dtype=bool)))


def normalized_psf_patch(
    psfs: np.ndarray, exposure_index: int, stamp_size: int
) -> np.ndarray:
    psf = np.asarray(psfs[:, :, 0, exposure_index], dtype=np.float32)
    psf_sum = float(psf.sum())
    if psf_sum > 0:
        psf = psf / psf_sum
    canvas = np.zeros((stamp_size, stamp_size), dtype=np.float32)
    h, w = psf.shape
    y0 = max((stamp_size - h) // 2, 0)
    x0 = max((stamp_size - w) // 2, 0)
    y1 = min(y0 + h, stamp_size)
    x1 = min(x0 + w, stamp_size)
    canvas[y0:y1, x0:x1] = psf[: y1 - y0, : x1 - x0]
    return canvas


def local_noise_sigma(
    variances: np.ndarray,
    center_y: int,
    center_x: int,
    exposure_index: int,
    *,
    window: int = 5,
) -> float:
    half = window // 2
    patch = np.asarray(
        variances[
            0,
            center_y - half : center_y + half + 1,
            center_x - half : center_x + half + 1,
            exposure_index,
        ],
        dtype=np.float32,
    )
    scale = float(np.sqrt(np.median(np.maximum(patch, 1e-8))))
    return max(scale, 1e-3)


def inject_transient(
    search_stamp: np.ndarray,
    psf_patch: np.ndarray,
    flux_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    injection = psf_patch * np.float32(flux_scale)
    return (search_stamp + injection).astype(np.float32), injection.astype(
        np.float32
    )


def xpois_difference(
    search_stamp: np.ndarray,
    template_stamp: np.ndarray,
    *,
    variance_stamp: np.ndarray | None = None,
    config: HscXPOISConfig | None = None,
) -> np.ndarray:
    (
        GaussianBasisComponent,
        solve_constant_kernel,
        solve_separable_kernel,
    ) = _import_xpois_symbols()
    cfg = config or HscXPOISConfig()
    if len(cfg.kernel_shape) != 2:
        raise ValueError("kernel_shape must contain exactly two values")
    if len(cfg.basis_sigmas) != len(cfg.basis_degrees):
        raise ValueError(
            "basis_sigmas and basis_degrees must have the same length"
        )
    if not cfg.basis_sigmas:
        raise ValueError("basis_sigmas must not be empty")

    reference = np.asarray(template_stamp, dtype=np.float64)
    target = np.asarray(search_stamp, dtype=np.float64)
    variance = None
    if variance_stamp is not None and cfg.use_variance:
        variance = np.asarray(variance_stamp, dtype=np.float64)
        if variance.shape != target.shape:
            raise ValueError("variance_stamp must match the stamp shape")

    components = [
        GaussianBasisComponent(float(sigma), int(degree))
        for sigma, degree in zip(
            cfg.basis_sigmas, cfg.basis_degrees, strict=True
        )
    ]
    if cfg.solver_mode == "constant":
        result = solve_constant_kernel(
            reference,
            target,
            components,
            kernel_shape=cfg.kernel_shape,
            variance=variance,
            background_degree=cfg.background_degree,
            flux_conserve=cfg.flux_conserve,
        )
    elif cfg.solver_mode == "separable":
        result = solve_separable_kernel(
            reference,
            target,
            components,
            kernel_shape=cfg.kernel_shape,
            variance=variance,
            background_degree=cfg.background_degree,
            flux_conserve=cfg.flux_conserve,
        )
    else:
        raise ValueError("solver_mode must be one of: constant, separable")
    return np.asarray(result.residual, dtype=np.float32)


def simple_difference(
    search_stamp: np.ndarray, template_stamp: np.ndarray
) -> np.ndarray:
    return (search_stamp - template_stamp).astype(np.float32)


def tile_group(center_y: int, center_x: int, tile_size: int) -> str:
    return f"{center_y // tile_size}_{center_x // tile_size}"


def assign_split(
    group: str, seed: int, fractions: tuple[float, float, float]
) -> int:
    digest = hashlib.sha1(f"{seed}:{group}".encode("utf-8")).digest()
    number = int.from_bytes(digest[:8], byteorder="big", signed=False)
    value = number / float(2**64 - 1)
    train_frac, val_frac, _ = fractions
    if value < train_frac:
        return SPLIT_TO_INDEX["train"]
    if value < train_frac + val_frac:
        return SPLIT_TO_INDEX["val"]
    return SPLIT_TO_INDEX["test"]


def prediction_group_indices(
    split: np.ndarray, split_name: str
) -> np.ndarray:
    split_index = SPLIT_TO_INDEX[split_name]
    return np.nonzero(np.asarray(split, dtype=np.int64) == split_index)[0]
