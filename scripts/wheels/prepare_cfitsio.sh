#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Run once per cibuildwheel Linux container, before the Python builds.
set -euo pipefail

cfitsio_prefix=${CUPHOTON_XDR_CFITSIO_ROOT:?Set the CFITSIO installation prefix}
cfitsio_commit=d6d27653740864a0483e74cff23c53e34e1dceee # CFITSIO 4.7.0
# GitHub may change archive compression; see "Stability of source code archives":
# https://docs.github.com/en/repositories/working-with-files/using-files/downloading-source-code-archives#stability-of-source-code-archives
cfitsio_sha256=b99c555309eee9187615a9d64cb9a7ad84c2f3e94a0241c11bb024ccd5c56dde
cfitsio_build_dir=$(mktemp -d)
trap 'rm -rf "$cfitsio_build_dir"' EXIT

cd "$cfitsio_build_dir"
curl --fail --location --retry 3 --proto '=https' --proto-redir '=https' \
  --output cfitsio.tar.gz \
  "https://codeload.github.com/HEASARC/cfitsio/tar.gz/${cfitsio_commit}"
printf '%s  cfitsio.tar.gz\n' "$cfitsio_sha256" | sha256sum --check --strict
tar -xzf cfitsio.tar.gz
cd "cfitsio-${cfitsio_commit}"
./configure --prefix="$cfitsio_prefix" \
  --enable-reentrant --disable-curl --without-bzip2 \
  --disable-static --enable-shared
make -j"${CUPHOTON_WHEEL_BUILD_JOBS:-2}"
make check
make install
install -Dm644 licenses/License.txt \
  "$cfitsio_prefix/share/licenses/cfitsio/License.txt"
