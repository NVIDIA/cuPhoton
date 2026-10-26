# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""The catalog's column contract: one schema for every item, empty or not,
scored or not.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

# name -> (dtype, stage, description)
BASE_COLUMNS: dict[str, tuple[str, str, str]] = {
    "candidate_id": ("object", "detect", "cu-<exposure_id>-<ccd_id>-<index>"),
    "x": (
        "int64",
        "detect",
        "candidate column (zero-based pixel of the rounded hotspot centroid)",
    ),
    "y": ("int64", "detect", "candidate row (zero-based)"),
    "peak_abs_sigma": (
        "float64",
        "detect",
        "peak |residual| / robust sigma within the hotspot",
    ),
    "pixel_count": (
        "int64",
        "detect",
        "pixels of the hotspot above threshold",
    ),
    "peak_value": (
        "float64",
        "detect",
        "residual at the peak pixel (image units)",
    ),
    "sign": ("int64", "detect", "+1 brightening, -1 fading"),
    "noise_match_factor": (
        "float32",
        "stamps",
        "scale applied to the stamp triplet before scoring (1 when disabled)",
    ),
}
XFIT_STATUS_COLUMNS: dict[str, tuple[str, str, str]] = {
    "converged": ("bool", "xfit", "Levenberg-Marquardt converged"),
    "chi_square": (
        "float64",
        "xfit",
        "chi-square of the dipole fit on the difference stamp",
    ),
    "reduced_chi_square": (
        "float64",
        "xfit",
        "chi-square per degree of freedom",
    ),
    "uncertainty_valid": ("bool", "xfit", "standard errors are usable"),
    "evaluations": ("int64", "xfit", "residual evaluations used"),
    "valid_pixel_count": ("int64", "xfit", "finite stamp pixels in the fit"),
}


def _position_columns(
    name: str, what: str
) -> dict[str, tuple[str, str, str]]:
    cols: dict[str, tuple[str, str, str]] = {}
    for axis in ("x", "y"):
        cols[f"{name}_{axis}"] = (
            "float64",
            "xfit",
            f"{what}, {axis} in pixels relative to the candidate's (x, y)",
        )
        cols[f"{name}_{axis}_err_fit"] = (
            "float64",
            "xfit",
            f"xFit covariance term of {name}_{axis}_err",
        )
        cols[f"{name}_{axis}_err"] = (
            "float64",
            "xfit",
            f"error bar of {name}_{axis} to quote (the terms listed in the "
            "README)",
        )
    cols[f"{name}_xy_cov"] = (
        "float64",
        "xfit",
        f"covariance of {name}_x and {name}_y (pixels squared)",
    )
    return cols


UNCERTAINTY_COLUMNS: dict[str, tuple[str, str, str]] = {
    "quality_flags": (
        "int64",
        "xfit",
        "bits of cuphoton.xfit.DipoleQualityFlag (0 when none is set)",
    ),
    **_position_columns("center", "midpoint of the two lobes"),
    **_position_columns("positive", "lobe with positive flux"),
    "separation": ("float64", "xfit", "distance between the lobes (pixels)"),
    "separation_err": (
        "float64",
        "xfit",
        "standard uncertainty of separation (NaN for coincident lobes)",
    ),
    "lobe_flux": (
        "float64",
        "xfit",
        "total flux of each lobe, 2 pi |amplitude| sigma_x sigma_y",
    ),
    "lobe_flux_err": ("float64", "xfit", "standard uncertainty of lobe_flux"),
    "psf_flux": (
        "float64",
        "xfit",
        "forced PSF flux at the source (NaN without a variance plane or a "
        "PSF width from xfit.psf_sigma_px or the refine stage's stars)",
    ),
    "psf_flux_err_fit": (
        "float64",
        "xfit",
        "xFit term of psf_flux_err, given the position and the PSF",
    ),
    "psf_flux_err": (
        "float64",
        "xfit",
        "error bar of psf_flux to quote (the terms listed in the README)",
    ),
}
CLASSIFY_COLUMNS: dict[str, tuple[str, str, str]] = {
    "logit": (
        "float32",
        "xscan",
        "classifier logit (NaN without a checkpoint)",
    ),
    "probability": (
        "float32",
        "xscan",
        "sigmoid(logit) (NaN without a checkpoint)",
    ),
    "decision": (
        "bool",
        "xscan",
        "probability >= classify.decision_threshold (false without a "
        "checkpoint)",
    ),
}
SKY_COLUMNS: dict[str, tuple[str, str, str]] = {
    "ra": (
        "float64",
        "terminal",
        "degrees, header WCS at the integer candidate pixel",
    ),
    "dec": (
        "float64",
        "terminal",
        "degrees, header WCS at the integer candidate pixel",
    ),
}
IDENTITY_COLUMNS: dict[str, tuple[str, str, str]] = {
    "exposure_id": (
        "object",
        "identity",
        "survey exposure identifier (Rubin: visit)",
    ),
    "ccd_id": (
        "object",
        "identity",
        "survey detector identifier (Rubin: detector)",
    ),
    "band": ("object", "identity", "filter band"),
    "astrometry": (
        "object",
        "identity",
        "how ra/dec were derived (the survey adapter's label)",
    ),
}


def xfit_columns(
    parameter_names: Sequence[str],
) -> dict[str, tuple[str, str, str]]:
    cols: dict[str, tuple[str, str, str]] = {}
    for name in parameter_names:
        cols[name] = ("float64", "xfit", f"dipole parameter {name}")
        cols[f"{name}_err"] = ("float64", "xfit", f"standard error of {name}")
    cols.update(XFIT_STATUS_COLUMNS)
    return cols


def schema(parameter_names: Sequence[str]) -> dict[str, tuple[str, str, str]]:
    return {
        **BASE_COLUMNS,
        **xfit_columns(parameter_names),
        **UNCERTAINTY_COLUMNS,
        **CLASSIFY_COLUMNS,
        **SKY_COLUMNS,
        **IDENTITY_COLUMNS,
    }


def empty_table(parameter_names: Sequence[str]) -> dict[str, np.ndarray]:
    """Zero-row columns with the production dtypes, so an empty item merges
    without promotion.
    """
    out: dict[str, np.ndarray] = {}
    for name, (dtype, _, _) in schema(parameter_names).items():
        if name in IDENTITY_COLUMNS:
            continue
        out[name] = np.empty(0, dtype=object if dtype == "object" else dtype)
    return out


def finish_table(
    table: Mapping[str, Any],
    n: int,
    identity_columns: Mapping[str, Any],
    astrometry: str,
    exposure_id: str,
    ccd_id: str,
    parameter_names: Sequence[str],
):
    """Coerce every column to its contract dtype and add identity columns;
    returns a DataFrame.
    """
    import pandas as pd

    sch = schema(parameter_names)
    cols: dict[str, Any] = {
        "candidate_id": np.array(
            [f"cu-{exposure_id}-{ccd_id}-{i:05d}" for i in range(n)],
            dtype=object,
        )
    }
    for name, (dtype, _, _) in sch.items():
        if name in IDENTITY_COLUMNS or name == "candidate_id":
            continue
        if name in table:
            values = np.asarray(table[name])
        elif name in CLASSIFY_COLUMNS or name.startswith("psf_flux"):
            values = (
                np.full(n, False)
                if dtype == "bool"
                else np.full(n, np.nan, dtype=dtype)
            )
        else:
            raise KeyError(
                f"catalog column {name} missing from the worker table"
            )
        cols[name] = values.astype(
            object if dtype == "object" else dtype, copy=False
        )
    df = pd.DataFrame(cols)
    for k, v in identity_columns.items():
        if k in sch and k not in {"exposure_id", "ccd_id", "band"}:
            raise ValueError(
                f"identity column {k!r} conflicts with the catalog schema"
            )
        df[k] = pd.Series([v] * n, dtype=object)
    df["astrometry"] = pd.Series([astrometry] * n, dtype=object)
    return df


def markdown_table(
    parameter_names: Sequence[str] = (
        "amplitude",
        "sigma_x",
        "sigma_y",
        "theta",
        "x_pos",
        "y_pos",
        "x_neg",
        "y_neg",
    ),
) -> str:
    rows = ["| column | dtype | stage | meaning |", "|---|---|---|---|"]
    for name, (dtype, stage, desc) in schema(parameter_names).items():
        rows.append(f"| `{name}` | {dtype} | {stage} | {desc} |")
    return "\n".join(rows)
