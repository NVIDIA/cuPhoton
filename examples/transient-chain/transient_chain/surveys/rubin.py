# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Rubin DP1 (LSSTComCam layout): the survey this example was built
on.

Visit files carry IMAGE, MASK and VARIANCE in HDUs 1, 2 and 3; template coadd
patches carry
IMAGE and MASK in HDUs 1 and 2; both use the LSST stack's ``MP_<NAME>``
mask-plane cards and
the hsc-masklite bad-plane policy. The template is the mean of every
overlapping coadd patch
reprojected onto the visit grid, unless the selection supplies
``template_fits`` already on
that grid. Every one of these facts is a parameter below.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from astropy.wcs import WCS

from ..fitsio import PlaneReader
from .base import (
    GridTemplate,
    Identity,
    ReprojectedSources,
    SurveyAdapter,
    TargetPlanes,
    bad_from_mask,
    resolve_mask_bits,
    variance_from,
)

DEFAULTS: dict[str, Any] = {
    "image_hdu": 1,
    "mask_hdu": 2,
    "variance_hdu": 3,
    "coadd_image_hdu": 1,
    "coadd_mask_hdu": 2,
    "mask_policy": "hsc-masklite",
    "variance_kind": "variance",
    "extra_keys": (
        "visit",
        "detector",
        "date",
        "tract",
        "patch",
        "physical_filter",
    ),
}


class RubinAdapter(SurveyAdapter):
    name = "rubin"
    astrometry_label = "header-wcs-approx"
    required_keys = ("visit", "detector", "band", "visit_image")

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        unknown = sorted(set(self.params) - set(DEFAULTS))
        if unknown:
            raise ValueError(
                f"survey.params for 'rubin' has unknown keys {unknown}; "
                f"known: {sorted(DEFAULTS)}"
            )
        self.p = {**DEFAULTS, **self.params}

    def validate_selection(self, sel, index):
        super().validate_selection(sel, index)
        if not sel.get("template_fits") and not sel.get("template_coadds"):
            raise ValueError(
                f"selection {index} needs 'template_coadds' (a list of coadd "
                f" patch files) or 'template_fits'"
            )

    def identity(self, sel):
        extra = {k: sel[k] for k in self.p["extra_keys"] if k in sel}
        return Identity(
            str(sel["visit"]), str(sel["detector"]), str(sel["band"]), extra
        )

    def input_paths(self, sel):
        paths = [sel["visit_image"]]
        if sel.get("template_fits"):
            paths.append(sel["template_fits"])
        else:
            paths.extend(sel["template_coadds"])
        return paths

    def payload(self, sel):
        keys = (
            "visit",
            "detector",
            "band",
            "visit_image",
            "template_coadds",
            "template_fits",
            *self.p["extra_keys"],
        )
        return {k: sel[k] for k in keys if k in sel}

    def load_target(self, sel, io: PlaneReader) -> TargetPlanes:
        p = self.p
        xp = io.xp
        hdus = [p["image_hdu"], p["mask_hdu"]] + (
            [p["variance_hdu"]] if p["variance_kind"] != "none" else []
        )
        arrays, _ = io.read(sel["visit_image"], hdus)
        image, mask = arrays[0], arrays[1]
        var = arrays[2] if len(arrays) > 2 else None
        from ..fitsio import inspect_plane

        header = inspect_plane(sel["visit_image"], p["image_hdu"]).header
        mask_header = inspect_plane(sel["visit_image"], p["mask_hdu"]).header
        bits = resolve_mask_bits(p["mask_policy"], mask_header)
        return TargetPlanes(
            image=image,
            bad=bad_from_mask(mask, bits, xp),
            variance=variance_from(var, p["variance_kind"], xp),
            wcs=WCS(header),
            header=header,
            mask_bits=bits,
            mask=mask,
        )

    def template(
        self,
        sel,
        target: TargetPlanes,
        io: PlaneReader,
        *,
        mapping_grid_step: int,
    ):
        from ..fitsio import inspect_plane

        p = self.p
        xp = io.xp
        override = sel.get("template_fits")
        if override:
            (img, mask), _ = io.read(
                override, [p["coadd_image_hdu"], p["coadd_mask_hdu"]]
            )
            bits = resolve_mask_bits(
                p["mask_policy"],
                inspect_plane(override, p["coadd_mask_hdu"]).header,
            )
            if tuple(img.shape) != target.shape:
                raise ValueError(
                    f"template_fits {override} has shape {tuple(img.shape)}, "
                    f" the visit has {target.shape}"
                )
            return GridTemplate(img, bad_from_mask(mask, bits, xp), xp)
        paths = list(sel["template_coadds"])
        planes, _ = io.read_batch(
            paths, [p["coadd_image_hdu"], p["coadd_mask_hdu"]]
        )
        sources = []
        for path, (img, mask) in zip(paths, planes, strict=True):
            header = inspect_plane(path, p["coadd_image_hdu"]).header
            bits = resolve_mask_bits(
                p["mask_policy"],
                inspect_plane(path, p["coadd_mask_hdu"]).header,
            )
            # bits are resolved per patch; the provider ORs raw bits, so keep
            # only the bad ones
            bad_bits_plane = mask if bits is None else (mask & bits)
            sources.append((img, bad_bits_plane, WCS(header)))
        return ReprojectedSources(
            sources,
            target.wcs,
            target.shape,
            xp,
            mapping_grid_step=mapping_grid_step,
        )
