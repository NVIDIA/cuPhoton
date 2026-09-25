#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Build and install the nvcomp batched-decompress pybind11 extension.
#
# Usage:  bash src/cuphoton/xdr/src/build.sh
# Env overrides: CUDA_HOME, PYTHON, VIRTUAL_ENV
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SRC_DIR/../../../.." && pwd)"
if ! command -v uv >/dev/null 2>&1; then
    echo "xdr build requires uv on PATH; install uv before building." >&2
    exit 1
fi

if [[ ${PYTHON+x} ]]; then
    if [[ -z "$PYTHON" ]] || ! PYTHON=$(command -v -- "$PYTHON"); then
        echo "PYTHON must name an executable Python interpreter." >&2
        exit 1
    fi
elif [[ -n ${VIRTUAL_ENV:-} ]]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON="$ROOT_DIR/.venv/bin/python"
else
    PYTHON=$(command -v python3 || command -v python || true)
fi
if [[ -z "$PYTHON" || ! -f "$PYTHON" || ! -x "$PYTHON" ]]; then
    echo "Python interpreter unavailable: ${PYTHON:-not found}; set PYTHON explicitly." >&2
    exit 1
fi
if [[ "$PYTHON" != /* ]]; then
    PYTHON="$PWD/$PYTHON"
fi

cd "$ROOT_DIR"
printf 'Building xdr with Python: %s\n' "$PYTHON"
uv pip install --python "$PYTHON" \
    'setuptools>=83.0.0' 'setuptools-scm==10.3.4' wheel 'pybind11>=3.0,<4'
exec env CUPHOTON_XDR_BUILD_EXT=1 uv pip install \
    --python "$PYTHON" \
    --no-build-isolation \
    -e "$ROOT_DIR[io]"
