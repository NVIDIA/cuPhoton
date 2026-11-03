# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import os
import sys
import sysconfig
from pathlib import Path

from setuptools import setup

ROOT = Path(__file__).resolve().parent
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _load_build_helpers(component):
    module_path = ROOT / "src" / "cuphoton" / component / "setup_package.py"
    spec = importlib.util.spec_from_file_location(
        f"_cuphoton_{component}_setup_package", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot load {component} build helpers: {module_path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _build_enabled(variable, *, default=False):
    mode = os.environ.get(variable, "1" if default else "0").strip().lower()
    if mode in _FALSE_VALUES:
        return False

    if mode not in _TRUE_VALUES:
        raise RuntimeError(
            f"{variable} must be one of: "
            + ", ".join(sorted(_FALSE_VALUES | _TRUE_VALUES))
        )

    return True


extensions = []
commands = {}
if _build_enabled("CUPHOTON_XDR_BUILD_EXT"):
    extensions.extend(_load_build_helpers("xdr").get_extensions())
if _build_enabled("CUPHOTON_XRAY_BUILD_EXT"):
    extensions.extend(_load_build_helpers("xray").get_extensions())
build_xfit = _build_enabled("CUPHOTON_XFIT_BUILD_EXT")
build_imaging = _build_enabled(
    "CUPHOTON_IMAGING_BUILD_EXT", default=build_xfit
)
if build_xfit or build_imaging:
    xfit = _load_build_helpers("xfit")
    commands["build_ext"] = xfit.CUDABuildExt
    if build_xfit:
        extensions.extend(xfit.get_extensions())
    if build_imaging:
        # Use the same selected SDK, architectures and licensed CCCL headers
        # for the independent imaging extension.
        imaging = xfit.get_extensions(require_cufft=True)[0]
        source = Path("src/cuphoton/core/src")
        imaging.name = "cuphoton.core._imaging_ext"
        imaging.sources = [str(source / "imaging_ext.cpp")]
        imaging.cuda_source_directory = source
        imaging.include_dirs.append(sysconfig.get_path("include"))
        imaging.libraries.append(":libcufft.so.12")
        units = (
            "imaging_copy",
            "imaging_selection",
            "imaging_reproject",
            "imaging_xpois",
            "imaging_spatial_xpois",
            "imaging_fits",
        )
        imaging.depends = [
            str(source / "imaging_api.h"),
            *(str(source / f"{unit}.cu") for unit in units),
        ]
        highest = int(imaging.architectures[-1].removesuffix("a"))
        portable_ptx = tuple(sorted({75, highest}))
        imaging.cuda_units = tuple(
            (unit, False, False, imaging.architectures, portable_ptx)
            for unit in units
        )
        extensions.append(imaging)

setup(ext_modules=extensions, cmdclass=commands)
