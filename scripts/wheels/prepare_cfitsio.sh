#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Run once per cibuildwheel Linux container, before the Python builds.
set -euo pipefail

cfitsio_prefix=${CUPHOTON_XDR_CFITSIO_ROOT:?Set the CFITSIO installation prefix}
cfitsio_version=4.7.0
cfitsio_sha256=f281ca292407d8682e83a37ea68fce6e1383d2449d923647ea62fa0570c59aee
cfitsio_build_dir=$(mktemp -d)
trap 'rm -rf "$cfitsio_build_dir"' EXIT

cd "$cfitsio_build_dir"
curl --fail --location --retry 3 --proto '=https' --proto-redir '=https' \
  --output cfitsio.tar.gz \
  "https://codeload.github.com/HEASARC/cfitsio/tar.gz/refs/tags/cfitsio-${cfitsio_version}"
printf '%s  cfitsio.tar.gz\n' "$cfitsio_sha256" | sha256sum --check --strict
tar -xzf cfitsio.tar.gz
cd "cfitsio-cfitsio-${cfitsio_version}"
./configure --prefix="$cfitsio_prefix" \
  --enable-reentrant --disable-curl --without-bzip2 \
  --disable-static --enable-shared
make -j"${CUPHOTON_WHEEL_BUILD_JOBS:-2}"
make check
make install
install -Dm644 licenses/License.txt \
  "$cfitsio_prefix/share/licenses/cfitsio/License.txt"
