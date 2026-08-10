# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
import subprocess
import sys
from dataclasses import fields

import pytest

from cuphoton import xfit, xpois, xrep
from cuphoton.core.cli import ComponentSpec
from cuphoton.xpois.ois import ConstantKernelFitResult
from cuphoton.xray.linear_prediction import LinearPredictionResult
from cuphoton.xrep.geometry import MaskedReprojectionResult, ReprojectionSpec
from cuphoton.xrep.mapping import PreparedReprojection
from cuphoton.xscan.dataset import StampDataset


def test_curated_root_exports_are_real_objects() -> None:
    assert xrep.PreparedReprojection is PreparedReprojection
    assert xrep.MaskedReprojectionResult is MaskedReprojectionResult
    assert callable(xrep.reproject_masked_array)
    assert callable(xrep.build_stack_spec_from_fits)
    assert callable(xpois.inspect_hsc_data_tree)
    assert callable(xpois.load_image_array)


def test_xfit_curated_exports_and_fit_signature() -> None:
    expected = {
        "BatchedLeastSquaresProblem",
        "DipoleFitResult",
        "GaussianDipoleModel",
        "LMConfig",
        "LMResult",
        "LMStatus",
        "StampDipoleModel",
        "batched_levenberg_marquardt",
        "fit_dipoles",
    }

    assert expected <= set(xfit.__all__)
    assert list(inspect.signature(xfit.fit_dipoles).parameters) == [
        "images",
        "model",
        "initial",
        "mask",
        "variance",
        "mode",
        "backend",
        "config",
    ]


def test_xrep_additions_do_not_expand_existing_dataclasses() -> None:
    assert [field.name for field in fields(ReprojectionSpec)] == [
        "mapping",
        "output_bbox",
        "source_origin",
        "interpolation",
        "mapping_grid_step",
        "area_scaling",
        "fill_value",
        "invalid_mask_value",
        "lanczos_a",
        "two_a_footprint",
    ]
    assert [field.name for field in fields(PreparedReprojection)] == [
        "x_local",
        "y_local",
        "relative_area",
        "valid",
    ]


def test_xrep_existing_function_signatures_remain_stable() -> None:
    assert list(inspect.signature(xrep.reproject_array).parameters) == [
        "source",
        "spec",
        "source_mask",
        "backend",
    ]
    assert list(
        inspect.signature(xrep.reproject_array_device).parameters
    ) == [
        "source",
        "spec",
        "source_mask",
        "backend",
        "prepared",
    ]
    assert (
        "variance_fill_value"
        in inspect.signature(xrep.reproject_masked_array).parameters
    )


@pytest.mark.parametrize(
    ("obj", "required_text"),
    [
        (ConstantKernelFitResult, "target - matched"),
        (xpois.SeparableKernelFitResult, "target - matched"),
        (PreparedReprojection, "d(source pixel)/d(destination pixel)"),
        (StampDataset, "do not alias the memory-mapped files"),
        (LinearPredictionResult, "radians per input time unit"),
        (ComponentSpec, "command-line metadata"),
    ],
)
def test_public_contract_docstrings_are_specific(
    obj: object,
    required_text: str,
) -> None:
    doc = inspect.getdoc(obj)
    assert doc is not None
    assert required_text in doc


@pytest.mark.parametrize(
    ("group", "description"),
    [
        (
            "xdr",
            "xDataReader: GPU-native FITS and HDF5 metadata loading.",
        ),
        (
            "xfit",
            "xFit: Batched nonlinear least-squares fitting for astronomical "
            "models.",
        ),
        (
            "xpois",
            "Optimal image subtraction for astronomical images.",
        ),
        (
            "xscan",
            "Transient-candidate dataset, training, and review workflows.",
        ),
        (
            "xrep",
            "xRep (xReproject): WCS-aware astronomical image reprojection "
            "and stacking.",
        ),
        ("xray", "GPU-accelerated X-ray detector analysis tools."),
    ],
)
def test_umbrella_component_groups(
    group: str,
    description: str,
) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "cuphoton", group, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert description in result.stdout


def test_cli_first_package_roots_do_not_import_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import cuphoton.xscan, cuphoton.xray; "
                "raise SystemExit('torch' in sys.modules)"
            ),
        ],
        check=False,
    )

    assert result.returncode == 0


@pytest.mark.parametrize(
    "command",
    [
        "doctor",
        "validation-viz",
        "linear-prediction-smoke",
        "detector-artifacts",
        "gpu-policy",
    ],
)
def test_xray_subcommands_have_descriptions(command: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cuphoton",
            "xray",
            "help",
            command,
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "\n\n" in result.stdout
    assert "usage:" in result.stdout
