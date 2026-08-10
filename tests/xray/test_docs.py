# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

DOCS = Path(__file__).resolve().parents[2] / "docs" / "xray"


def test_public_docs_cover_environment_and_gpu_policy():
    environment = (DOCS / "ENVIRONMENT.md").read_text(encoding="utf-8")
    gpu_first = (DOCS / "GPU-FIRST.md").read_text(encoding="utf-8")

    assert "make test-xray" in environment
    assert (
        "uv sync --locked --extra dev --extra gpu --extra viz" in environment
    )
    assert "uv run cuphoton xray doctor" in environment
    assert "uv run xray " not in environment
    assert "GPU-first" in gpu_first
    assert "NumPy paths" in gpu_first
    assert "concrete CuPy or NumPy" in gpu_first


def test_public_docs_cover_validation_and_distributed_artifacts():
    validation = (DOCS / "VALIDATION-VIZ.md").read_text(encoding="utf-8")
    distributed = (DOCS / "DISTRIBUTED-DETECTOR-ARTIFACTS.md").read_text(
        encoding="utf-8"
    )

    assert "validation-viz" in validation
    assert "workflow-viz" in validation
    assert "phonon-viz" in validation
    assert "detector-artifact-distributed" in distributed
    assert "executor dry-run" in distributed
