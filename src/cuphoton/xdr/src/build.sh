#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Build and install the nvcomp batched-decompress pybind11 extension.
#
# Usage:  bash src/cuphoton/xdr/src/build.sh
# Env overrides: CUDA_HOME, PYTHON (default .venv/bin/python)
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(cd "$SRC_DIR/../../../.." && pwd)"
PYTHON=${PYTHON:-"$ROOT_DIR/.venv/bin/python"}
if [[ ! -x "$PYTHON" ]]; then
    PYTHON=python3
fi

cd "$ROOT_DIR"
exec env CUPHOTON_XDR_BUILD_EXT=1 uv pip install \
    --python "$PYTHON" \
    --no-build-isolation \
    -e "$ROOT_DIR[gpu]"
