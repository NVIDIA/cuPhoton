# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import os
import sys
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


def _build_enabled(variable):
    mode = os.environ.get(variable, "0").strip().lower()
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
if _build_enabled("CUPHOTON_XFIT_BUILD_EXT"):
    xfit = _load_build_helpers("xfit")
    extensions.extend(xfit.get_extensions())
    commands["build_ext"] = xfit.CUDABuildExt

setup(ext_modules=extensions, cmdclass=commands)
