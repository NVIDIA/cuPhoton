# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest

from cuphoton.ximage.adapters.legacy import legacy_masks_to_good_pixels


@pytest.mark.parametrize("dilation", [1, 2, 3, 4])
def test_legacy_conversion_covers_batches_and_frames_without_mutation(
    dilation,
):
    masks = np.zeros((2, 7, 7, 12, 2), dtype=np.uint8)
    masks[:, 3, 3, 10, :] = 1
    original = masks.copy()
    conservative = legacy_masks_to_good_pixels(
        masks, dilation_factor=dilation
    )
    expected = np.ones((2, 7, 7, 2), dtype=bool)
    start = 3 - (dilation - 1) // 2
    stop = 4 + dilation // 2
    expected[:, start:stop, start:stop, :] = False
    assert conservative.dtype == np.dtype(bool)
    np.testing.assert_array_equal(conservative, expected)
    np.testing.assert_array_equal(
        legacy_masks_to_good_pixels(masks, policy="non_conservative"),
        np.ones_like(expected),
    )
    np.testing.assert_array_equal(masks, original)


def test_legacy_policies_map_camera_planes_to_good_pixels():
    masks = np.eye(12, dtype=np.uint8).reshape(1, 1, 12, 12, 1)
    conservative = legacy_masks_to_good_pixels(masks)
    non_conservative = legacy_masks_to_good_pixels(
        masks, policy="non_conservative"
    )
    expected = np.ones((1, 1, 12, 1), dtype=bool)
    expected[:, :, [0, 4, 8], :] = False
    np.testing.assert_array_equal(non_conservative, expected)
    expected[:, :, [1, 7, 9, 10], :] = False
    np.testing.assert_array_equal(conservative, expected)


@pytest.mark.parametrize(
    "shape", [(1, 4, 4, 2), (1, 4, 4, 11, 2), (1, 4, 4, 13, 2)]
)
def test_legacy_conversion_rejects_wrong_shape(shape):
    with pytest.raises(ValueError, match="Legacy masks must have shape"):
        legacy_masks_to_good_pixels(np.zeros(shape, dtype=bool))


@pytest.mark.parametrize("dilation", [0, -1, 1.5, True])
def test_legacy_conversion_rejects_invalid_dilation(dilation):
    with pytest.raises(
        ValueError, match="dilation_factor must be a positive integer"
    ):
        legacy_masks_to_good_pixels(
            np.zeros((1, 4, 4, 12, 2)), dilation_factor=dilation
        )


def test_legacy_conversion_rejects_unknown_policy():
    with pytest.raises(ValueError, match="Unsupported legacy mask policy"):
        legacy_masks_to_good_pixels(
            np.zeros((1, 4, 4, 12, 2)), policy="unknown"
        )
