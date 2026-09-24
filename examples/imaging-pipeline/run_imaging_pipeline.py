#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Load, align, subtract, fit, and plot a synthetic pair of FITS images.

Three static stars and one moving source illustrate cuPhoton's imaging APIs.

Run from the repository root:

    uv sync --locked --extra gpu --extra viz
    bash src/cuphoton/xdr/src/build.sh
    uv run --locked --extra gpu --extra viz python \
        examples/imaging-pipeline/run_imaging_pipeline.py

The full walkthrough requires Linux and a CUDA 13-capable NVIDIA GPU.
The xDR load step allocates a CuPy batch and has no CPU fallback.
See docs/components/xdr.md for the native extension's CUDA headers and
thread-safe CFITSIO build prerequisites.

If WORK_DIR is unset, outputs go to ./imaging-pipeline-output (gitignored).
The notebook of the same sections is
examples/imaging-pipeline/run_imaging_pipeline.ipynb.

"""

# === SETUP ===
import os
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image
from scipy.ndimage import gaussian_filter
from scipy.ndimage import shift as ndshift

from cuphoton.xdr import batch_to_device
from cuphoton.xfit import GaussianDipoleModel, fit_dipoles
from cuphoton.xpois import GaussianBasisComponent, solve_constant_kernel
from cuphoton.xrep import (
    BBox,
    Grid,
    build_stack_spec_from_fits,
    make_north_up_wcs,
    reproject_stack,
)

try:
    from IPython.display import display
except ImportError:

    def display(obj):
        pass


WORK = Path(
    os.environ.get("WORK_DIR", Path.cwd() / "imaging-pipeline-output")
)
(WORK / "fits").mkdir(parents=True, exist_ok=True)
(WORK / "figs").mkdir(parents=True, exist_ok=True)
SHAPE = (256, 256)
STARS = (
    (70.0, 80.0, 45.0, 2.10),
    (175.0, 190.0, 32.0, 2.40),
    (40.0, 200.0, 22.0, 2.00),
)
MOVER_OLD = (120.0, 108.0, 14.0, 2.10)  # y, x, amp, sigma
MOVER_NEW = (126.5, 116.0, 14.0, 2.10)
CRVAL = (150.0, 2.0)
PIXEL_SCALE = 0.2  # arcsec / pixel
SCIENCE_SHIFT = (20.0, -32.0)  # dy, dx; large enough to see by eye
SEEING_SIGMA = 0.65  # extra Gaussian blur applied to the science frame


def gaussian2d(shape, y0, x0, amp, sigma):
    y, x = np.mgrid[: shape[0], : shape[1]]
    return amp * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma**2))


def make_scene(moving, seeing=0.0):
    image = np.full(SHAPE, 0.45, dtype=np.float32)
    for y0, x0, amp, sigma in STARS:
        image += gaussian2d(SHAPE, y0, x0, amp, sigma).astype(np.float32)
    y0, x0, amp, sigma = moving
    image += gaussian2d(SHAPE, y0, x0, amp, sigma).astype(np.float32)
    if seeing:
        image = gaussian_filter(image, sigma=seeing).astype(np.float32)
    return image


def write_fits(path, data, wcs):
    # batch_to_device defaults to HDU index 1 (first image extension).
    header = wcs.to_header()
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(
                data=np.asarray(data, np.float32), name="IMAGE", header=header
            ),
        ]
    ).writeto(path, overwrite=True)


def stretch_gray(arr):
    a = np.asarray(arr, np.float64)
    lo, hi = np.nanpercentile(a, (1, 99.5))
    x = np.clip((a - lo) / (hi - lo + 1e-8), 0, 1)
    x = np.where(np.isfinite(a), x, 0)
    return (x * 255).astype(np.uint8)


def stretch_div(arr, limit=None):
    a = np.asarray(arr, np.float64)
    if limit is None:
        limit = float(np.nanpercentile(np.abs(a), 99))
    m = max(limit, 1e-8)
    t = np.clip(a / m, -1, 1)
    rgb = np.stack(
        [
            np.where(t >= 0, 1.0, 1.0 + t),
            1.0 - np.abs(t),
            np.where(t <= 0, 1.0, 1.0 - t),
        ],
        axis=-1,
    )
    rgb = np.where(np.isfinite(a)[..., None], rgb, 0)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def panel(arr, diverging=False, size=256, limit=None):
    data = stretch_div(arr, limit=limit) if diverging else stretch_gray(arr)
    mode = "RGB" if diverging else "L"
    im = Image.fromarray(data, mode=mode).convert("RGB")
    return im.resize((size, size), Image.NEAREST)


def panel_kernel(arr, size=256, zoom=10):
    """Zoom a small kernel by an integer factor and center it."""
    k = np.asarray(arr)
    im = Image.fromarray(stretch_gray(k), mode="L").convert("RGB")
    im = im.resize((k.shape[1] * zoom, k.shape[0] * zoom), Image.NEAREST)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    x = (size - im.width) // 2
    y = (size - im.height) // 2
    canvas.paste(im, (x, y))
    return canvas


def show_row(*ims, path=None, gap=16):
    tiles = [np.array(im.convert("RGB")) for im in ims]
    h, w = tiles[0].shape[:2]
    canvas = Image.new(
        "RGB",
        (len(tiles) * w + (len(tiles) - 1) * gap, h),
        (255, 255, 255),
    )
    x = 0
    for tile in tiles:
        canvas.paste(Image.fromarray(tile), (x, 0))
        x += w + gap
    if path is not None:
        canvas.save(path)
        print("wrote", path)
    display(canvas)


wcs_t = make_north_up_wcs(CRVAL, SHAPE, PIXEL_SCALE)
wcs_s = make_north_up_wcs(CRVAL, SHAPE, PIXEL_SCALE)
dx, dy = SCIENCE_SHIFT[1], SCIENCE_SHIFT[0]
wcs_s.wcs.crpix = [
    float(wcs_s.wcs.crpix[0]) + dx,
    float(wcs_s.wcs.crpix[1]) + dy,
]
wcs_s.wcs.set()

template = make_scene(MOVER_OLD)
science = ndshift(
    make_scene(MOVER_NEW, seeing=SEEING_SIGMA),
    shift=SCIENCE_SHIFT,
    order=3,
    mode="constant",
    cval=0.45,
).astype(np.float32)
write_fits(WORK / "fits" / "template.fits", template, wcs_t)
write_fits(WORK / "fits" / "science.fits", science, wcs_s)

# === LOAD ===
(images,) = batch_to_device(
    [WORK / "fits" / "template.fits", WORK / "fits" / "science.fits"],
    hdu_indices=(1,),
)
print(images.shape, images.dtype, images.device)

# === xRep ===
native, spec = build_stack_spec_from_fits(
    [WORK / "fits" / "template.fits", WORK / "fits" / "science.fits"],
    hdu=1,
    grid=Grid.from_wcs(wcs_t),
    output_bbox=BBox(0, 0, SHAPE[1], SHAPE[0]),
    interpolation="lanczos3",
    mapping_grid_step=16,
)
aligned = reproject_stack(native, spec, backend="auto")
print(aligned.images.shape, aligned.backend)
reference, target = aligned.images

# === xPois ===
# Fit only the static scene, excluding both mover positions and their wings.
fit_mask = np.ones(SHAPE, dtype=bool)
fit_mask[105:142, 93:132] = False
fit = solve_constant_kernel(
    reference,
    target,
    [GaussianBasisComponent(sigma=SEEING_SIGMA, degree=0)],
    kernel_shape=(15, 15),
    background_degree=0,
    flux_conserve=True,
    fit_mask=fit_mask,
    backend="auto",
)
print(
    f"kernel backend: {fit.backend}; sum: {float(fit.kernel.sum()):.8f}; "
    f"fit pixels: {int(fit.fit_pixel_count)}"
)
# No variance is supplied, so chi2 is an unweighted sum of squares here.
print("static-scene residual RMS:", np.sqrt(fit.chi2 / fit.fit_pixel_count))

# === xFit ===
STAMP = 21
half = STAMP // 2
cy, cx = 123, 112  # midpoint of the planted mover
stamp = fit.residual[cy - half : cy + half + 1, cx - half : cx + half + 1]

# Seed the fit from the observed stamp extrema, in centered x/y coordinates.
y_pos, x_pos = np.unravel_index(np.argmax(stamp), stamp.shape)
y_neg, x_neg = np.unravel_index(np.argmin(stamp), stamp.shape)
initial = np.array(
    [
        [
            stamp.max(),
            2.0,
            2.0,
            0.0,
            x_pos - half,
            y_pos - half,
            x_neg - half,
            y_neg - half,
        ]
    ]
)

model = GaussianDipoleModel((STAMP, STAMP), dtype=np.float64)
result = fit_dipoles(
    stamp[None], model=model, initial=initial, backend="auto"
)
print(
    f"dipole backend: {result.backend}; device: {result.device}; "
    f"converged: {bool(result.converged[0])}"
)
for name, start, fitted in zip(
    model.parameter_names, initial[0], result.parameters[0]
):
    # Orientation is unconstrained for these nearly circular lobes.
    if name != "theta":
        print(f"{name:10s}  initial={start:7.3f}  fit={fitted:7.3f}")

# Use the planted scene only to check the recovered fit.
expected_sigma = np.hypot(MOVER_NEW[3], SEEING_SIGMA)
print(
    f"expected sigma: {expected_sigma:.6f}; "
    f"fitted sigma_x/sigma_y: {result.parameters[0, 1:3]}"
)
stamp_peak = float(np.max(np.abs(stamp)))
print(
    "max dipole residual / stamp peak:",
    float(np.max(np.abs(result.residuals[0]))) / stamp_peak,
)
# Planted centers in stamp coordinates: x_pos, y_pos, x_neg, y_neg.
truth_xy = np.array(
    [
        MOVER_NEW[1] - cx,
        MOVER_NEW[0] - cy,
        MOVER_OLD[1] - cx,
        MOVER_OLD[0] - cy,
    ]
)
print("planted centers (x_pos, y_pos, x_neg, y_neg):", truth_xy)
print("center errors (pixels):", result.parameters[0, 4:] - truth_xy)

# === PLOT ===
host = images.get()
model_img = np.asarray(model.evaluate(result.parameters, mode="difference"))[
    0
]
show_row(
    panel(host[0]), panel(host[1]), path=WORK / "figs" / "01_loaded_fits.png"
)
show_row(
    panel(reference),
    panel(target),
    path=WORK / "figs" / "02_xrep_aligned.png",
)
show_row(
    panel_kernel(fit.kernel),
    panel(fit.matched),
    panel(fit.residual, diverging=True),
    path=WORK / "figs" / "03_subtraction.png",
)
show_row(
    panel(stamp, diverging=True, limit=stamp_peak),
    panel(model_img, diverging=True, limit=stamp_peak),
    # xFit returns model minus data; display data minus model, as in xPois.
    panel(-result.residuals[0], diverging=True, limit=stamp_peak),
    path=WORK / "figs" / "04_dipole.png",
)
