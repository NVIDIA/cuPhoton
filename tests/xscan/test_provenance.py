# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the hardened training-label provenance guard.

Each case is a concrete fail-open probe the previous guard let through.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from cuphoton.xscan.training import check_training_label_provenance


def _dataset(tmp_path, labels, rows) -> object:
    directory = tmp_path / "ds"
    directory.mkdir()
    np.save(directory / "labels.npy", np.asarray(labels))
    if rows is not None:
        (directory / "metadata.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n",
            encoding="utf-8",
        )
    return directory


def test_clean_dataset_is_ok(tmp_path) -> None:
    directory = _dataset(
        tmp_path,
        [0, 1],
        [
            {"label": 0, "label_source": "reviewed"},
            {"label": 1, "label_source": "reviewed"},
        ],
    )
    provenance = check_training_label_provenance(directory)
    assert provenance["ok"] is True
    assert provenance["errors"] == []


def test_absent_metadata_flagged(tmp_path) -> None:
    directory = _dataset(tmp_path, [0, 1], None)
    provenance = check_training_label_provenance(directory)
    assert provenance["ok"] is False
    assert "metadata_incomplete" in provenance["errors"]


def test_non_binary_labels_flagged(tmp_path) -> None:
    directory = _dataset(
        tmp_path,
        [0, 1, 2],
        [{"label": v, "label_source": "s"} for v in (0, 1, 2)],
    )
    provenance = check_training_label_provenance(directory)
    assert provenance["labels_binary"] is False
    assert "labels_not_binary" in provenance["errors"]


def test_float_labels_flagged(tmp_path) -> None:
    directory = _dataset(
        tmp_path,
        [0.9, 1.1],
        [{"label": 1, "label_source": "s"} for _ in range(2)],
    )
    provenance = check_training_label_provenance(directory)
    assert "labels_not_binary" in provenance["errors"]


def test_metadata_label_mismatch_flagged(tmp_path) -> None:
    # metadata says [1, 0]; labels.npy says [0, 1].
    directory = _dataset(
        tmp_path,
        [0, 1],
        [
            {"label": 1, "label_source": "s"},
            {"label": 0, "label_source": "s"},
        ],
    )
    provenance = check_training_label_provenance(directory)
    assert "metadata_label_mismatch" in provenance["errors"]


def test_target_label_unavailable_flagged(tmp_path) -> None:
    directory = _dataset(
        tmp_path,
        [0, 1],
        [
            {
                "label": 0,
                "label_source": "s",
                "target_label_available": False,
            },
            {"label": 1, "label_source": "s", "target_label_available": True},
        ],
    )
    provenance = check_training_label_provenance(directory)
    assert "target_label_unavailable" in provenance["errors"]


@pytest.mark.parametrize(
    "source",
    [
        None,
        "",
        "missing",
        "  ",
        "null",
        False,
        123,
        "unlabeled",
        "  unlabeled_lsstcomcam_smoke_placeholder  ",
    ],
)
def test_untrusted_source_flagged(tmp_path, source) -> None:
    directory = _dataset(
        tmp_path,
        [0, 1],
        [
            {"label": 0, "label_source": source},
            {"label": 1, "label_source": "reviewed"},
        ],
    )
    provenance = check_training_label_provenance(directory)
    assert "label_source_untrusted" in provenance["errors"]


def test_metadata_label_missing_flagged(tmp_path) -> None:
    directory = _dataset(
        tmp_path,
        [0, 1],
        [
            {"label_source": "s"},  # no "label" field
            {"label": 1, "label_source": "s"},
        ],
    )
    provenance = check_training_label_provenance(directory)
    assert "metadata_label_missing" in provenance["errors"]


def test_metadata_label_not_binary_flagged(tmp_path) -> None:
    # Rounding must NOT hide 0.49/1.49 metadata labels.
    directory = _dataset(
        tmp_path,
        [0, 1],
        [
            {"label": 0.49, "label_source": "s"},
            {"label": 1.49, "label_source": "s"},
        ],
    )
    provenance = check_training_label_provenance(directory)
    assert "metadata_label_not_binary" in provenance["errors"]


def test_target_available_invalid_type_flagged(tmp_path) -> None:
    # A string "false" is truthy; treat it as invalid, not available.
    directory = _dataset(
        tmp_path,
        [0, 1],
        [
            {
                "label": 0,
                "label_source": "s",
                "target_label_available": "false",
            },
            {"label": 1, "label_source": "s", "target_label_available": True},
        ],
    )
    provenance = check_training_label_provenance(directory)
    assert "target_label_available_invalid" in provenance["errors"]
