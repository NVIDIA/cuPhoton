#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# cibuildwheel puts the current build interpreter on PATH.
set -euo pipefail

wheel_script_dir=$(cd "$(dirname "$0")" && pwd)
uv pip install --python "$(command -v python)" --only-binary :all: \
  --requirement "$wheel_script_dir/build-requirements.txt"
