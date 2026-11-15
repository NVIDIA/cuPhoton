# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-candidate, exact-stamp xFit feature investigation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def fit_feature_bundle(dataset, output, *, backend="cupy"):
    """Use the native exporter, fit CLI and feature correspondence guards.

    The caller must retain the shared GPU lock for CUDA backends. Inputs are
    explicit generated datasets only, never demo fixtures. Failed/invalid
    fits remain represented by the native feature gates.
    """
    from cuphoton.xscan.xfit_features import (
        build_xfit_feature_bundle,
        export_xfit_input,
        load_xfit_feature_matrix,
    )

    dataset, output = Path(dataset), Path(output)
    if output.exists():
        raise ValueError("xFit investigation output must not exist")
    variance = dataset / "variance.npy"
    if not variance.is_file():
        raise ValueError("variance plane required for this investigation")
    output.mkdir(parents=True)
    archive, run, bundle = (
        output / "input.npz",
        output / "fit",
        output / "features",
    )
    exported = export_xfit_input(
        dataset_dir=dataset,
        output_path=archive,
        variance_path=variance,
        image_unit="normalized_electron",
    )
    with (
        (output / "fit.stdout").open("w") as stdout,
        (output / "fit.stderr").open("w") as stderr,
    ):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "cuphoton",
                "xfit",
                "fit-dipoles",
                "--input",
                str(archive),
                "--output-dir",
                str(run),
                "--model",
                "gaussian",
                "--mode",
                "difference",
                "--backend",
                backend,
                "--compute-dtype",
                "float64",
            ],
            stdout=stdout,
            stderr=stderr,
            check=True,
        )
    build_xfit_feature_bundle(
        dataset_dir=dataset,
        xfit_run_dir=run,
        output_dir=bundle,
        missing_policy="indicator",
    )
    matrix = load_xfit_feature_matrix(dataset_dir=dataset, feature_dir=bundle)
    names = list(matrix.feature_names)
    receipt = dict(
        rows=len(matrix.values),
        backend=backend,
        source_verification=exported["source_hash_verification"],
        native_join_diagnostics=matrix.join_diagnostics,
    )
    for name in ("fit_present", "fit_valid", "uncertainty_valid"):
        values = matrix.values[:, names.index(name)]
        receipt[name] = dict(
            available=int(np.sum(values > 0.5)),
            unavailable=int(np.sum(values <= 0.5)),
            fraction=float(np.mean(values > 0.5)),
        )
    (output / "coverage.json").write_text(
        json.dumps(receipt, indent=2) + "\n"
    )
    return bundle, receipt
