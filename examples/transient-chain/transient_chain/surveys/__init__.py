# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Survey adapters: ``rubin`` (one survey's layout as shipped) and ``generic``
(any FITS pair by parameters). ``survey.name`` picks one; it has no default.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .base import Identity, SurveyAdapter, TargetPlanes, TemplateProvider
from .generic import GenericFitsAdapter
from .rubin import RubinAdapter

ADAPTERS: dict[str, type[SurveyAdapter]] = {
    RubinAdapter.name: RubinAdapter,
    GenericFitsAdapter.name: GenericFitsAdapter,
}


def make_adapter(survey_cfg: Mapping[str, Any]) -> SurveyAdapter:
    name = survey_cfg.get("name")
    if name is None:
        raise ValueError(
            f"config 'survey.name' is required; available: {sorted(ADAPTERS)}"
        )
    if name not in ADAPTERS:
        raise ValueError(
            f"unknown survey '{name}'; available: {sorted(ADAPTERS)}"
        )
    return ADAPTERS[name](survey_cfg.get("params") or {})


__all__ = [
    "ADAPTERS",
    "GenericFitsAdapter",
    "Identity",
    "RubinAdapter",
    "SurveyAdapter",
    "TargetPlanes",
    "TemplateProvider",
    "make_adapter",
]
