# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Rubin image, catalog, and reprojection adapters."""

from .image import (
    LSST_IMAGES_VERSION,
    RUBIN_IMAGE_FORMATS,
    RubinImageBounds,
    RubinImageDependencyError,
    RubinImageError,
    RubinImageFormatError,
    RubinImageGeometry,
    RubinImageView,
    inspect_rubin_image_geometry,
    local_pixel_is_addressable,
    local_pixel_is_inside,
    local_to_absolute_pixel,
    open_rubin_image,
    read_rubin_image_bounds,
)
from .warp import (
    RubinMaskedWarpResult,
    RubinWarpDependencyError,
    RubinWarpPlanes,
    rubin_view_to_warp_planes,
    warp_rubin_image_view,
)

__all__ = [
    "LSST_IMAGES_VERSION",
    "RUBIN_IMAGE_FORMATS",
    "RubinImageBounds",
    "RubinImageDependencyError",
    "RubinImageError",
    "RubinImageFormatError",
    "RubinImageGeometry",
    "RubinImageView",
    "RubinMaskedWarpResult",
    "RubinWarpDependencyError",
    "RubinWarpPlanes",
    "inspect_rubin_image_geometry",
    "local_pixel_is_addressable",
    "local_pixel_is_inside",
    "local_to_absolute_pixel",
    "open_rubin_image",
    "read_rubin_image_bounds",
    "rubin_view_to_warp_planes",
    "warp_rubin_image_view",
]
