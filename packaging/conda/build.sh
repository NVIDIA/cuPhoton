#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export CUPHOTON_XDR_BUILD_EXT=1
export CUPHOTON_XDR_NATIVE_PREFIX="$PREFIX"
export CUPHOTON_XFIT_BUILD_EXT=1
export CUDA_HOME="$PREFIX"
export CUDACXX="$BUILD_PREFIX/bin/nvcc"
export CUPHOTON_XFIT_CCCL_ROOT="$SP_DIR/cupy/_core/include/cupy/_cccl"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_CUPHOTON="$PKG_VERSION"
"$PYTHON" -m pip install . --no-deps --no-build-isolation --no-index -v
