# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""A centroid control distinguishes equal-depth, differently located dips."""

import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from blend import measure_case  # noqa: E402
from pipeline import write_exposure  # noqa: E402
from scene import add_star, draw_exposure, normalized_psf  # noqa: E402


@pytest.mark.gpu
def test_gpu_localizes_fading_neighbor_despite_similar_aperture_depth(
    tmp_path,
):
    cp = pytest.importorskip("cupy")
    if not cp.cuda.runtime.getDeviceCount():
        pytest.skip("CUDA device required")
    yy, xx = np.indices((25, 25))
    psf = normalized_psf(np.exp(-((xx - 12) ** 2 + (yy - 12) ** 2) / 8), 1.2)
    base = np.zeros((256, 256))
    target = np.array([128.2, 127.7])
    neighbor = target + [1.5, 0.7]
    for point in ([48, 48], [208, 48], [48, 208]):
        add_star(base, psf, point, 100_000)
    add_star(base, psf, target, 100_000)
    add_star(base, psf, neighbor, 5_000)
    rng = np.random.default_rng(31415)
    measured = []
    for index, source in enumerate((target, neighbor)):
        dimmed = base.copy()
        add_star(dimmed, psf, source, -1800)
        paths = []
        for state, model in (("out", base), ("in", dimmed)):
            data, variance = draw_exposure(model, 316, 0.562, rng, 16)
            path = tmp_path / f"{index}-{state}.fits"
            write_exposure(
                path,
                {
                    "science": data,
                    "variance": variance,
                    "mask": np.zeros(data.shape, np.uint8),
                    "header": fits.Header({"SIMULATE": True}),
                },
            )
            paths.append(path)
        # No injection truth file or PSF truth is supplied to measurement.
        measured.append(measure_case(*paths, target))
    planet, blend = measured
    assert planet["backend"] == blend["backend"] == "cupy"
    assert np.linalg.norm(planet["fading_source_offset_xy_pixels"]) < 0.6
    assert np.linalg.norm(blend["fading_source_offset_xy_pixels"]) > 1.0
    assert abs(
        planet["aperture_depth"] - blend["aperture_depth"]
    ) < 3 * np.hypot(
        planet["aperture_depth_error"], blend["aperture_depth_error"]
    )
