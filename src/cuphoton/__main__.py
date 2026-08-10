# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Module entry point for ``python -m cuphoton``."""

from __future__ import annotations

from cuphoton.core.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
