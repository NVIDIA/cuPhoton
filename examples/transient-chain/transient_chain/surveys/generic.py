# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Any FITS survey described by parameters: which file and HDU hold each
plane.

``survey.params`` for ``name: generic``::

    identity: {exposure_id: expnum, ccd_id: ccdnum, band: band}   # selection
    keys to read
    extra_keys: [field, filter_name]                              # copied
    into the catalog
    target:
      path_key: image                 # selection key holding the science
      image path
      image_hdu: 0                    # index, EXTNAME, or null for the first
      2D image
      mask_path_key: null             # another file for the mask, or null =
      same file
      mask_hdu: null                  # with no mask_path_key: no mask plane;
      with one: its first 2D image
      variance_path_key: null
      variance_hdu: null              # null = no variance
      variance_kind: variance         # variance | inverse_variance | none
    template:
      path_key: template
      image_hdu: 0
      mask_path_key: null
      mask_hdu: null
      on_target_grid: false           # true: same pixel grid; false:
      reproject through WCS
    mask_policy: strict               # none | strict | hsc-masklite |
    {planes: [...]} | {bits: N}

Defaults match a plain single-extension image pair with no mask and no
variance.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from astropy.wcs import WCS

from ..fitsio import PlaneReader, hdu_index, inspect_plane
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

_TARGET = {
    "path_key": "image",
    "image_hdu": None,
    "mask_path_key": None,
    "mask_hdu": None,
    "variance_path_key": None,
    "variance_hdu": None,
    "variance_kind": "variance",
}
_TEMPLATE = {
    "path_key": "template",
    "image_hdu": None,
    "mask_path_key": None,
    "mask_hdu": None,
    "on_target_grid": False,
}
_IDENTITY = {"exposure_id": "exposure_id", "ccd_id": "ccd_id", "band": "band"}


def _merge(
    name: str, defaults: Mapping[str, Any], given: Mapping[str, Any] | None
) -> dict[str, Any]:
    given = dict(given or {})
    unknown = sorted(set(given) - set(defaults))
    if unknown:
        raise ValueError(
            f"survey.params.{name} has unknown keys {unknown}; known: "
            f"{sorted(defaults)}"
        )
    return {**defaults, **given}


class GenericFitsAdapter(SurveyAdapter):
    name = "generic"
    astrometry_label = "header-wcs"

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        super().__init__(params)
        known = {
            "identity",
            "extra_keys",
            "target",
            "template",
            "mask_policy",
        }
        unknown = sorted(set(self.params) - known)
        if unknown:
            raise ValueError(
                f"survey.params for 'generic' has unknown keys {unknown}; "
                f"known: {sorted(known)}"
            )
        self.id_keys = _merge(
            "identity", _IDENTITY, self.params.get("identity")
        )
        self.extra_keys = tuple(self.params.get("extra_keys") or ())
        self.target_p = _merge("target", _TARGET, self.params.get("target"))
        self.template_p = _merge(
            "template", _TEMPLATE, self.params.get("template")
        )
        self.mask_policy = self.params.get("mask_policy", "strict")
        t = self.target_p
        keys = [
            self.id_keys["exposure_id"],
            self.id_keys["ccd_id"],
            self.id_keys["band"],
            t["path_key"],
            self.template_p["path_key"],
        ]
        for k in (
            t["mask_path_key"],
            t["variance_path_key"],
            self.template_p["mask_path_key"],
        ):
            if k:
                keys.append(k)
        self.required_keys = tuple(dict.fromkeys(keys))

    def identity(self, sel):
        extra = {k: sel[k] for k in self.extra_keys if k in sel}
        return Identity(
            str(sel[self.id_keys["exposure_id"]]),
            str(sel[self.id_keys["ccd_id"]]),
            str(sel[self.id_keys["band"]]),
            extra,
        )

    def input_paths(self, sel):
        t, tp = self.target_p, self.template_p
        keys = [
            t["path_key"],
            t["mask_path_key"],
            t["variance_path_key"],
            tp["path_key"],
            tp["mask_path_key"],
        ]
        return list(dict.fromkeys(sel[k] for k in keys if k and k in sel))

    def payload(self, sel):
        return {
            k: sel[k]
            for k in (*self.required_keys, *self.extra_keys)
            if k in sel
        }

    def _read_role(self, io: PlaneReader, sel, path_key, hdu, same_file_path):
        """Read one plane by (file, hdu).

        No path key and no hdu: the role is absent, returns (None, None). A
        path key alone means the
        first 2D image of that file; an hdu alone means that HDU of the same
        file as the image.
        """
        if hdu is None and not path_key:
            return None, None
        path = sel[path_key] if path_key else same_file_path
        index = hdu_index(path, hdu)
        (arr,), _ = io.read(path, [index])
        return arr, inspect_plane(path, index).header

    def load_target(self, sel, io: PlaneReader) -> TargetPlanes:
        t = self.target_p
        xp = io.xp
        path = sel[t["path_key"]]
        image_hdu = hdu_index(path, t["image_hdu"])
        (image,), _ = io.read(path, [image_hdu])
        header = inspect_plane(path, image_hdu).header
        mask, mask_header = self._read_role(
            io, sel, t["mask_path_key"], t["mask_hdu"], path
        )
        var, _ = self._read_role(
            io, sel, t["variance_path_key"], t["variance_hdu"], path
        )
        bits = (
            None
            if mask is None
            else resolve_mask_bits(self.mask_policy, mask_header)
        )
        bad = None if mask is None else bad_from_mask(mask, bits, xp)
        return TargetPlanes(
            image=image,
            bad=bad,
            variance=variance_from(var, t["variance_kind"], xp),
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
        tp = self.template_p
        xp = io.xp
        path = sel[tp["path_key"]]
        image_hdu = hdu_index(path, tp["image_hdu"])
        (image,), _ = io.read(path, [image_hdu])
        header = inspect_plane(path, image_hdu).header
        mask, mask_header = self._read_role(
            io, sel, tp["mask_path_key"], tp["mask_hdu"], path
        )
        bits = (
            None
            if mask is None
            else resolve_mask_bits(self.mask_policy, mask_header)
        )
        if tp["on_target_grid"]:
            if tuple(image.shape) != target.shape:
                raise ValueError(
                    f"template {path} has shape {tuple(image.shape)}, the "
                    f"target has {target.shape}; "
                    "set template.on_target_grid: false to reproject through "
                    "the WCS"
                )
            return GridTemplate(
                image,
                None if mask is None else bad_from_mask(mask, bits, xp),
                xp,
            )
        bad_bits_plane = (
            None
            if mask is None
            else (mask if bits is None else (mask & bits))
        )
        return ReprojectedSources(
            [(image, bad_bits_plane, WCS(header))],
            target.wcs,
            target.shape,
            xp,
            mapping_grid_step=mapping_grid_step,
        )
