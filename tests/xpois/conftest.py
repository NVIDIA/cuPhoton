# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from cuphoton.xpois import ois


@pytest.fixture(autouse=True)
def deterministic_auto_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep default-backend tests independent of the test host's GPUs."""

    monkeypatch.setattr(
        ois,
        "_backend_available",
        lambda backend: backend == "cpu",
    )
