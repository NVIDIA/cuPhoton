# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Convert the legacy twelve-plane camera mask to good-pixel masks."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.ndimage import maximum_filter

from cuphoton.ximage.data import XImageArray


def legacy_masks_to_good_pixels(
    masks: XImageArray,
    *,
    policy: Literal["conservative", "non_conservative"] = "conservative",
    dilation_factor: int = 1,
) -> np.ndarray:
    """Convert ``(batch, y, x, 12, frame)`` masks to boolean valid pixels.

    Both policies exclude chip gaps and borders in planes 0, 4, and 8.
    The conservative policy also excludes stripe planes 1, 7, and 9 and
    large-source plane 10, dilated by a ``dilation_factor``-pixel square.
    The input is preserved. Apply this adapter before storing a bundle.
    """

    if policy not in ("conservative", "non_conservative"):
        raise ValueError(f"Unsupported legacy mask policy: {policy}")
    if masks.ndim != 5 or masks.shape[3] != 12:
        raise ValueError(
            "Legacy masks must have shape (batch, y, x, 12, frame)"
        )
    if (
        isinstance(dilation_factor, (bool, np.bool_))
        or not isinstance(dilation_factor, (int, np.integer))
        or dilation_factor <= 0
    ):
        raise ValueError("dilation_factor must be a positive integer")

    planes = np.asarray(masks, dtype=bool)
    excluded = np.any(planes[:, :, :, [0, 4, 8], :], axis=3)
    if policy == "conservative":
        excluded |= np.any(planes[:, :, :, [1, 7, 9], :], axis=3)
        excluded |= maximum_filter(
            planes[:, :, :, 10, :],
            size=(1, dilation_factor, dilation_factor, 1),
            mode="constant",
            cval=False,
        )
    return ~excluded
