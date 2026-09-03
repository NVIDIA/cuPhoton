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
from cuphoton.xpois.noise import (
    ConstantKernelNoiseResult,
    standardize_constant_kernel_residual,
)
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
    assert callable(xpois.solve_spatial_als)
    assert xpois.SpatialALSConfig().spatial_degree == 2
    assert xpois.ConstantKernelNoiseResult is ConstantKernelNoiseResult
    assert (
        xpois.standardize_constant_kernel_residual
        is standardize_constant_kernel_residual
    )


def test_xpois_noise_signature_is_explicit() -> None:
    assert list(
        inspect.signature(
            xpois.standardize_constant_kernel_residual
        ).parameters
    ) == [
        "residual",
        "kernel",
        "target_variance",
        "reference_variance",
        "valid_mask",
    ]


def test_xfit_curated_exports_and_fit_signature() -> None:
    expected = {
        "BatchedLeastSquaresProblem",
        "DeviceDipoleFitResult",
        "DipoleFitResult",
        "DipoleFitUncertaintyReason",
        "GaussianDipoleModel",
        "JacobianFunction",
        "LMConfig",
        "LMResult",
        "LMStatus",
        "NormalEquationsFunction",
        "ResidualFunction",
        "StampDipoleModel",
        "batched_levenberg_marquardt",
        "fit_dipoles",
        "fit_dipoles_device",
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
    assert list(inspect.signature(xfit.fit_dipoles_device).parameters) == [
        "images",
        "model",
        "initial",
        "mask",
        "variance",
        "mode",
        "config",
    ]
    assert [field.name for field in fields(xfit.DipoleFitResult)] == [
        "parameters",
        "parameter_names",
        "status",
        "converged",
        "evaluations",
        "residual_norm",
        "chi_square",
        "valid_pixel_count",
        "valid_pixel_fraction",
        "null_chi_square",
        "delta_chi_square",
        "fractional_null_improvement",
        "degrees_of_freedom",
        "reduced_chi_square",
        "covariance",
        "standard_errors",
        "uncertainty_valid",
        "uncertainty_reason",
        "residuals",
        "backend",
        "device",
        "dtype",
        "model",
        "mode",
    ]


def test_xpois_device_api_is_additive_and_legacy_signature_is_stable() -> (
    None
):
    expected = {
        "ConstantKernelFitResult",
        "DeviceConstantKernelFitResult",
        "solve_constant_kernel",
        "solve_constant_kernel_device",
    }

    assert expected <= set(xpois.__all__)
    assert list(
        inspect.signature(xpois.solve_constant_kernel).parameters
    ) == [
        "reference",
        "target",
        "components",
        "kernel_shape",
        "variance",
        "fit_mask",
        "background_degree",
        "flux_conserve",
        "flux_reference_index",
        "backend",
    ]
    assert list(
        inspect.signature(xpois.solve_constant_kernel_device).parameters
    ) == [
        "reference",
        "target",
        "components",
        "kernel_shape",
        "variance",
        "fit_mask",
        "background_degree",
        "flux_conserve",
        "flux_reference_index",
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
        (ConstantKernelNoiseResult, "marginal diagonal"),
        (xpois.SeparableKernelFitResult, "target - matched"),
        (xpois.SpatialALSFitResult, "target - matched"),
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
            "xDataReader: GPU-native FITS loading.",
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


def test_xpois_device_api_does_not_eagerly_import_cupy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from cuphoton import xpois; "
                "assert callable(xpois.solve_constant_kernel_device); "
                "raise SystemExit('cupy' in sys.modules)"
            ),
        ],
        check=False,
    )

    assert result.returncode == 0


def test_xfit_device_api_does_not_eagerly_import_cupy() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from cuphoton import xfit; "
                "assert callable(xfit.fit_dipoles_device); "
                "raise SystemExit('cupy' in sys.modules)"
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
