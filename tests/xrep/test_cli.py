# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import numpy as np
from astropy.io import fits

from cuphoton.core.cli import get_component, run_component
from cuphoton.xrep import make_north_up_wcs


def _run_cli(argv: list[str]) -> int:
    return run_component("xrep", argv)


def _write_test_fits(path: Path) -> Path:
    data = np.arange(25, dtype=np.float64).reshape(5, 5)
    wcs = make_north_up_wcs(
        (150.0, 2.0),
        shape=data.shape,
        pixel_scale_arcsec=0.2,
    )
    fits.PrimaryHDU(
        data=data.astype(np.float32),
        header=wcs.to_header(),
    ).writeto(path, overwrite=True)
    return path


def test_component_registry_uses_canonical_namespace() -> None:
    spec = get_component("xrep")

    assert spec.group == "xrep"
    assert spec.import_name == "cuphoton.xrep"


def test_group_program_name_is_used_in_help_output(capsys) -> None:
    rc = _run_cli(["help", "reproject-image"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xrep reproject-image" in captured.out
    assert "--input" in captured.out
    assert "-c FILE, --conf=FILE" not in captured.out


def test_help_for_reproject_stack_command(capsys) -> None:
    rc = _run_cli(["help", "reproject-stack"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xrep reproject-stack" in captured.out
    assert "--inputs" in captured.out
    assert "--mapping-mode" not in captured.out


def test_help_for_benchmark_reproject_image_command(capsys) -> None:
    rc = _run_cli(["help", "benchmark-reproject-image"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xrep benchmark-reproject-image" in captured.out
    assert "--repeats" in captured.out
    assert "--warmup" in captured.out


def test_help_for_benchmark_backend_variants_command(capsys) -> None:
    rc = _run_cli(["help", "benchmark-backend-variants"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xrep benchmark-backend-variants" in captured.out
    assert "--variants" in captured.out
    assert "--mask-cases" in captured.out


def test_help_for_compare_backends_command(capsys) -> None:
    rc = _run_cli(["help", "compare-backends"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xrep compare-backends" in captured.out
    assert "--backends" in captured.out
    assert "--reference-backend" in captured.out


def test_reproject_image_help_documents_explicit_auto_backend(capsys) -> None:
    rc = _run_cli(["help", "reproject-image"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "auto" in captured.out


def test_inspect_image_command_emits_json(tmp_path: Path, capsys) -> None:
    fits_path = _write_test_fits(tmp_path / "image.fits")

    rc = _run_cli(["inspect-image", "--input", str(fits_path), "--hdu", "0"])
    captured = capsys.readouterr()

    assert rc == 0
    assert str(fits_path) in captured.out
