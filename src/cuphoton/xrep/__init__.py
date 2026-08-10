# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""WCS-aware astronomical image reprojection and stacking."""

from cuphoton import __version__

from .geometry import (
    BBox,
    DeviceReprojectionResult,
    Grid,
    MaskedReprojectionResult,
    ReprojectionResult,
    ReprojectionSpec,
    StackReprojectionResult,
    StackReprojectionSpec,
    bbox_union,
    bbox_wcs,
    make_north_up_wcs,
)
from .mapping import (
    PreparedReprojection,
    estimate_source_bbox_on_grid,
    make_grid_mapping,
    make_wcs_mapping,
    make_wcs_pair_transform,
    make_wcs_pair_transform_with_offset,
    prepare_reprojection,
)
from .reproject import (
    build_stack_spec_from_fits,
    reproject_array,
    reproject_array_device,
    reproject_fits,
    reproject_masked_array,
    reproject_stack,
)

__all__ = [
    "__version__",
    "BBox",
    "DeviceReprojectionResult",
    "Grid",
    "MaskedReprojectionResult",
    "PreparedReprojection",
    "ReprojectionResult",
    "ReprojectionSpec",
    "StackReprojectionResult",
    "StackReprojectionSpec",
    "bbox_union",
    "bbox_wcs",
    "build_stack_spec_from_fits",
    "estimate_source_bbox_on_grid",
    "make_grid_mapping",
    "make_north_up_wcs",
    "make_wcs_mapping",
    "make_wcs_pair_transform",
    "make_wcs_pair_transform_with_offset",
    "prepare_reprojection",
    "reproject_array",
    "reproject_array_device",
    "reproject_fits",
    "reproject_masked_array",
    "reproject_stack",
]
