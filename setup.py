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


def _load_xdr_build_helpers():
    module_path = ROOT / "src" / "cuphoton" / "xdr" / "setup_package.py"
    spec = importlib.util.spec_from_file_location(
        "_cuphoton_xdr_setup_package", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Cannot load xDataReader build helpers: {module_path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _get_ext_modules():
    mode = os.environ.get("CUPHOTON_XDR_BUILD_EXT", "auto").lower()
    if mode in _FALSE_VALUES:
        return []

    try:
        return _load_xdr_build_helpers().get_extensions()
    except Exception as exc:
        if mode in _TRUE_VALUES:
            raise
        print(
            "WARNING: xDataReader native extension build skipped: "
            f"{type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return []


setup(ext_modules=_get_ext_modules())
