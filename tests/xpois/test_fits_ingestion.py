# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Reader integration preserves xPois input and executor contracts."""

import builtins

import numpy as np
from astropy.io import fits

from cuphoton.xpois import data, workflows
from cuphoton.xpois.batch import BatchFitOptions
from cuphoton.xpois.ois import GaussianBasisComponent


def test_selected_planes_preserve_mask_bits_wcs_and_reader_receipts(tmp_path):
    path = tmp_path / "planes.fits"
    image = np.arange(30, dtype=np.float32).reshape(5, 6)
    mask = np.zeros(image.shape, dtype=np.int64)
    mask[2, 3] = 1 << 40
    mask_hdu = fits.ImageHDU(mask, name="MASK")
    mask_hdu.header["MP_BAD"] = 40
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(image, name="IMAGE"),
            mask_hdu,
            fits.ImageHDU(image + 1, name="VARIANCE"),
        ]
    ).writeto(path)
    receipts = []
    loaded, _, used = data.load_image_with_wcs(
        path, fits_reader="astropy", read_metadata=receipts
    )
    variance, _, variance_hdu = data.load_variance_with_wcs(
        path, fits_reader="astropy", read_metadata=receipts
    )
    loaded_mask, used_mask, planes = data.load_mask_with_planes(
        path, fits_reader="astropy", read_metadata=receipts
    )
    assert (used, variance_hdu, used_mask) == (1, 3, 2)
    np.testing.assert_array_equal(loaded, image)
    np.testing.assert_array_equal(variance, image + 1)
    np.testing.assert_array_equal(loaded_mask, mask)
    assert planes == {"BAD": 40}
    assert len(receipts) == 3
    assert all(row["reader"] == "astropy" for row in receipts)


def test_cpu_workflow_auto_reader_never_imports_gpu(tmp_path, monkeypatch):
    image = np.random.default_rng(8).normal(size=(13, 15))
    path = tmp_path / "image.fits"
    fits.PrimaryHDU(image).writeto(path)
    imported = builtins.__import__
    gpu_imports = []

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"cupy", "kvikio"}:
            gpu_imports.append(name)
            raise AssertionError("CPU workflow must not load the GPU reader")
        return imported(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)
    result = workflows.run_constant_kernel_fit(
        reference_path=path,
        target_path=path,
        output_root=tmp_path / "runs",
        name="cpu-fits",
        reference_hdu=None,
        target_hdu=None,
        kernel_shape=(3, 3),
        components=[GaussianBasisComponent(sigma=1, degree=0)],
        variance_path=None,
        fit_mask_path=None,
        background_degree=0,
        flux_conserve=True,
        backend="cpu",
        review=False,
    )
    assert gpu_imports == []
    assert result.summary["fits_reader"] == "auto"
    assert all(
        row["reader"] == "astropy" for row in result.summary["fits_reads"]
    )


def test_fits_reader_survives_distributed_options_round_trip():
    options = BatchFitOptions(
        kernel_shape=(3, 3),
        basis_sigmas=(1,),
        basis_degrees=(0,),
        fits_reader="xdr",
        xdr_options={"postprocess": "separate"},
    )
    assert BatchFitOptions.from_payload(options.to_payload()) == options
