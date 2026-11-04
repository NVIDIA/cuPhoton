# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Pinned archive data and controlled image pairs for the Dragon notebook."""

from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import gaussian_filter

SOURCE_INFO = json.loads(
    Path(__file__).with_name("source-data.json").read_text()
)
KERNEL_SIGMA = 1.5
KERNEL_RADIUS = 7
STAMP_SIZE = 51
FOOTPRINT_RADIUS = STAMP_SIZE // 2 + KERNEL_RADIUS
SPLIT_X = {
    "train": (72, 144, 216, 288),
    "val": (392, 464, 536, 608),
    "test": (712, 784, 856, 928),
}
GROUP_Y = (96, 208, 320, 432, 544, 656, 768, 880)


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify_source(path: Path) -> None:
    if path.stat().st_size != SOURCE_INFO["bytes"]:
        raise ValueError(f"Wrong archive file size: {path}")
    if _sha256(path) != SOURCE_INFO["sha256"]:
        raise ValueError(f"Archive SHA-256 mismatch: {path}")


def acquire_image(output_root: Path, local_file: Path | None = None) -> Path:
    """Return the verified cached mosaic, downloading only when needed.

    Pass ``local_file`` to stage the pinned bytes before an offline session.
    An existing cache is verified on every call. A failed transfer never
    becomes the cached source file.
    """
    destination = (
        Path(output_root).resolve()
        / "data"
        / "source"
        / SOURCE_INFO["filename"]
    )
    if destination.exists():
        _verify_source(destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".fits.part")
    try:
        if local_file is not None:
            source = Path(local_file).resolve()
            _verify_source(source)
            shutil.copyfile(source, partial)
        else:
            request = urllib.request.Request(
                SOURCE_INFO["url"],
                headers={"User-Agent": "cuPhoton-Dragon-example"},
            )
            with (
                urllib.request.urlopen(request, timeout=45) as response,
                partial.open("wb") as stream,
            ):
                shutil.copyfileobj(response, stream)
        _verify_source(partial)
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)
    return destination


def _write_image(
    path: Path,
    array: np.ndarray,
    header: fits.Header,
    *,
    compressed: bool = False,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        plane = fits.CompImageHDU(
            array,
            header=header,
            name="SCI",
            compression_type="GZIP_1",
            quantize_level=0,
            tile_shape=(128, 128),
        )
    else:
        plane = fits.ImageHDU(array, header=header, name="SCI")
    fits.HDUList([fits.PrimaryHDU(), plane]).writeto(path)
    return str(path.resolve())


def _add_source(
    image: np.ndarray,
    x: float,
    y: float,
    flux: float,
    sigma_x: float,
    sigma_y: float,
) -> None:
    """Add a unit-integral Gaussian within a bounded pixel footprint."""
    radius = int(np.ceil(6 * max(sigma_x, sigma_y)))
    x0, x1 = int(np.floor(x)) - radius, int(np.floor(x)) + radius + 1
    y0, y1 = int(np.floor(y)) - radius, int(np.floor(y)) + radius + 1
    yy, xx = np.mgrid[y0:y1, x0:x1]
    profile = np.exp(
        -0.5 * (((xx - x) / sigma_x) ** 2 + ((yy - y) / sigma_y) ** 2)
    )
    image[y0:y1, x0:x1] += flux * profile / profile.sum()


def _scene_arrays(
    background: np.ndarray,
    rng: np.random.Generator,
    split: str,
    item_id: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    reference = background.astype(np.float64, copy=True)
    target_scene = reference.copy()
    coordinates = [(x, y) for y in GROUP_Y for x in SPLIT_X[split]]
    order = rng.permutation(len(coordinates))
    candidates = []
    artifacts = []
    fit_mask = np.ones(background.shape, dtype=np.uint8)
    radius = FOOTPRINT_RADIUS
    fit_mask[:radius] = fit_mask[-radius:] = 0
    fit_mask[:, :radius] = fit_mask[:, -radius:] = 0
    for index, coordinate_index in enumerate(order):
        x, y = coordinates[coordinate_index]
        group = f"m51-x{x:04d}-y{y:04d}"
        truth: dict[str, Any] = {}
        if index < 16:
            kind, label = "moving_source", 1
            label_source = "synthetic_moving_source"
            flux = float(rng.uniform(8, 14))
            dx = float(rng.uniform(3, 5))
            dy = float(rng.uniform(-3, 3))
            x_old, y_old = x - dx / 2, y - dy / 2
            x_new, y_new = x + dx / 2, y + dy / 2
            _add_source(reference, x_old, y_old, flux, 2, 2)
            _add_source(target_scene, x_new, y_new, flux, 2, 2)
            truth = {
                "reference_xy": [x_old, y_old],
                "target_xy": [x_new, y_new],
                "motion_xy": [dx, dy],
                "separation_pixels": float(np.hypot(dx, dy)),
                "integrated_flux": flux,
                "reference_sigma_pixels": 2.0,
                "matched_sigma_pixels": 2.5,
            }
        elif index < 24:
            kind, label = "detector_artifact", 0
            label_source = "synthetic_detector_artifact"
            shape = "sharp" if index % 2 == 0 else "short_track"
            flux = float(rng.uniform(8, 14))
            artifacts.append((x, y, flux, shape))
            truth = {"shape": shape, "integrated_flux": flux}
        else:
            kind, label = "unchanged_control", 0
            label_source = "synthetic_background_control"
        fit_mask[y - radius : y + radius + 1, x - radius : x + radius + 1] = 0
        candidates.append(
            {
                "candidate_id": f"{item_id}-c{index:02d}",
                "x": x,
                "y": y,
                "label": label,
                "label_source": label_source,
                "target_label_available": True,
                "split": split,
                "split_group": group,
                "kind": kind,
                "truth": truth,
            }
        )
    target = gaussian_filter(
        target_scene, KERNEL_SIGMA, radius=KERNEL_RADIUS, mode="reflect"
    )
    for x, y, flux, shape in artifacts:
        sigma_x = 0.45 if shape == "sharp" else 4.5
        _add_source(target, x, y, flux, sigma_x, 0.45)
    noise_sigma = float(rng.uniform(0.008, 0.012))
    target += rng.normal(0, noise_sigma, target.shape)
    variance = np.full(target.shape, noise_sigma**2, dtype=np.float32)
    return (
        reference.astype(np.float32),
        target.astype(np.float32),
        variance,
        fit_mask,
        candidates,
    )


def _write_scene(
    directory: Path,
    background: np.ndarray,
    header: fits.Header,
    rng: np.random.Generator,
    split: str,
    item_id: str,
    *,
    compressed: bool,
) -> dict:
    reference, target, variance, mask, candidates = _scene_arrays(
        background, rng, split, item_id
    )
    variance_header = header.copy()
    variance_header["BUNIT"] = "(MJy/sr)^2"
    mask_header = header.copy()
    del mask_header["BUNIT"]
    mask_header["MASKTRUE"] = (1, "Pixel can be used for kernel fitting")
    result = {
        "item_id": item_id,
        "reference": _write_image(
            directory / "reference.fits", reference, header
        ),
        "target": _write_image(directory / "target.fits", target, header),
        "variance": _write_image(
            directory / "variance.fits", variance, variance_header
        ),
        "fit_mask": _write_image(
            directory / "fit-mask.fits", mask, mask_header
        ),
        "hdu": 1,
        "split": split,
        "candidates": candidates,
        "representative_candidate_id": candidates[0]["candidate_id"],
        "target_noise_sigma": float(np.sqrt(variance[0, 0])),
        "kernel_sigma_pixels": KERNEL_SIGMA,
        "kernel_size": 2 * KERNEL_RADIUS + 1,
    }
    if compressed:
        result["target_compressed"] = _write_image(
            directory / "target-gzip.fits",
            target,
            header,
            compressed=True,
        )
    return result


def _reprojection_example(pair: dict) -> dict:
    """Translate pixels and their WCS together for a real alignment step."""
    target_path = Path(pair["target"])
    with fits.open(target_path) as hdus:
        target = hdus[1].data.copy()
        header = hdus[1].header.copy()
    # raw[y + dy, x + dx] and target[y, x] represent the same sky position.
    dx, dy = 3, -2
    raw = np.zeros_like(target)
    raw[:dy, dx:] = target[-dy:, :-dx]
    valid = np.zeros(target.shape, dtype=np.uint8)
    valid[:dy, dx:] = 1
    header["CRPIX1"] += dx
    header["CRPIX2"] += dy
    raw_path = _write_image(
        target_path.with_name("raw-target.fits"), raw, header
    )
    mask_header = header.copy()
    del mask_header["BUNIT"]
    mask_path = _write_image(
        target_path.with_name("raw-target-valid.fits"), valid, mask_header
    )
    pair["raw_target"] = raw_path
    return {
        "raw_target": raw_path,
        "raw_valid_mask": mask_path,
        "target": pair["target"],
        "reference": pair["reference"],
        "hdu": 1,
        "pixel_offset": {"dx": dx, "dy": dy},
        "valid_mask_true_means": "observed pixel",
        "description": (
            "Integer pixel translation with a matching WCS change. "
            "Align this image to the reference grid; subtraction uses "
            "the separately prepared aligned target."
        ),
    }


def prepare_images(
    output_root: Path,
    source_path: Path,
    seed: int = 2026,
    images: int = 8,
) -> dict:
    """Write aligned scenes, injection truth and a spatially split corpus.

    There are 320/96/96 training/validation/test candidates. A split owns
    fixed, non-overlapping sky footprints; noise and labels vary between
    scenes. Presentation scenes use test footprints and independent random
    streams. This measures a controlled task on related M51 backgrounds,
    not generalization to independent survey observations.
    """
    if images < 1:
        raise ValueError("images must be positive")
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    source_path = Path(source_path).resolve()
    _verify_source(source_path)
    directory = Path(output_root).resolve() / "data"
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        metadata = json.loads(manifest_path.read_text())
        if (
            metadata["seed"] != seed
            or len(metadata["pairs"]) != images
            or metadata["source"]["sha256"] != SOURCE_INFO["sha256"]
        ):
            raise ValueError(
                "Existing data configuration differs; use a new output root"
            )
        for filename, digest in metadata["files"].items():
            path = Path(filename)
            if not path.is_file() or _sha256(path) != digest:
                raise ValueError(
                    f"Prepared FITS file missing or changed: {path}"
                )
        return metadata
    if any(
        (directory / name).exists()
        for name in ("archive-crop.fits", "corpus", "pairs")
    ):
        raise ValueError(
            "Incomplete prepared data preserved; use a new output root"
        )
    y_bounds, x_bounds = SOURCE_INFO["crop_yx"]
    crop = (slice(*y_bounds), slice(*x_bounds))
    with fits.open(source_path) as hdus:
        original = hdus[SOURCE_INFO["hdu"]]
        if list(original.data.shape) != SOURCE_INFO["shape"]:
            raise ValueError("Unexpected archive image dimensions")
        background = original.data[crop].astype(np.float32)
        header = WCS(original.header).slice(crop).to_header()
    if background.shape != (1024, 1024) or not np.isfinite(background).all():
        raise ValueError(
            "The pinned 1024-square crop must be entirely finite"
        )
    header["BUNIT"] = SOURCE_INFO["units"]
    header["OBJECT"] = "M51 controlled injection demo"
    header["ORIGIN"] = "Spitzer/SINGS via NASA/IPAC IRSA"
    header["SIMULATE"] = (False, "Unmodified pixels cropped from the archive")
    original_crop = _write_image(
        directory / "archive-crop.fits", background, header
    )
    header["SIMULATE"] = (True, "Modified archive scene, not observed epochs")
    corpus = []
    for split_index, (split, count) in enumerate(
        (("train", 10), ("val", 3), ("test", 3))
    ):
        for index in range(count):
            item_id = f"{split}-{index:02d}"
            rng = np.random.default_rng([seed, 1, split_index, index])
            corpus.append(
                _write_scene(
                    directory / "corpus" / item_id,
                    background,
                    header,
                    rng,
                    split,
                    item_id,
                    compressed=False,
                )
            )
    pairs = []
    for index in range(images):
        item_id = f"demo-{index:02d}"
        rng = np.random.default_rng([seed, 2, index])
        pairs.append(
            _write_scene(
                directory / "pairs" / item_id,
                background,
                header,
                rng,
                "test",
                item_id,
                compressed=True,
            )
        )
    metadata = {
        "source": {**SOURCE_INFO, "path": str(source_path)},
        "crop": {"path": original_crop, "hdu": 1, "shape": [1024, 1024]},
        "seed": seed,
        "stamp_size": STAMP_SIZE,
        "footprint_radius": FOOTPRINT_RADIUS,
        "corpus": corpus,
        "pairs": pairs,
        "reprojection": _reprojection_example(pairs[0]),
        "simulation": {
            "units": SOURCE_INFO["units"],
            "noise": "Independent additive target Gaussian noise",
            "reference": "Fixed noisy archive scene plus injected sources",
            "positive_label": "Injected PSF-like moving source",
            "negative_label": "Detector artifact or unchanged control",
            "presentation_split": "test",
            "scope": "Controlled simulations on related archival backgrounds",
        },
    }
    owned_paths = {str(source_path), original_crop}
    for scene in corpus + pairs:
        owned_paths.update(
            scene[key]
            for key in (
                "reference",
                "target",
                "variance",
                "fit_mask",
                "target_compressed",
                "raw_target",
            )
            if key in scene
        )
    owned_paths.add(metadata["reprojection"]["raw_valid_mask"])
    metadata["files"] = {
        name: _sha256(Path(name)) for name in sorted(owned_paths)
    }
    partial_manifest = manifest_path.with_suffix(".json.part")
    partial_manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    partial_manifest.replace(manifest_path)
    return metadata
