# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from cuphoton import _photometry
from cuphoton.core.cli import ApplicationContext, get_component


def test_registered_component_uses_shared_application_context(
    tmp_path,
) -> None:
    component = get_component("ximage")

    context = ApplicationContext.for_component(
        component,
        environ={"HOME": str(tmp_path)},
    )

    assert context.component is component
    assert context.runs_dir == (
        tmp_path / ".local" / "state" / "cuphoton" / "ximage" / "runs"
    )


def test_shared_photometry_contract_remains_public() -> None:
    assert callable(_photometry.detect_sources)
    assert callable(_photometry.elliptical_aperture_flux)
