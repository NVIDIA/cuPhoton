# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""What the chain needs to know about a survey.

A ``SurveyAdapter`` turns one selection row into device-resident target
planes, a template
provider, and an identity record. The chain downstream of these three calls is
survey-free:
it applies bad masks, subtracts, detects, cuts stamps, fits, scores and writes
a catalog with
the generic identity columns ``exposure_id``, ``ccd_id``, ``band`` plus
whatever extras the
adapter supplies.

Mask policies are resolved here so every adapter speaks the same vocabulary:

- ``none``: no bad pixels from the mask plane (non-finite pixels are still
  bad)
- ``strict``: any nonzero mask bit is bad
- ``hsc-masklite`` (alias ``masklite``): the LSST stack's nine hsc-masklite
  planes, read from
  ``MP_<NAME>`` header cards; the plane list is cuPhoton's
- ``{"planes": [...]}``: named planes read from ``MP_`` cards
- ``{"bits": <int>}``: a literal bitmask
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from ..fitsio import PlaneReader

try:
    from cuphoton.xpois.workflows import (
        _HSC_MASKLITE_PLANES as HSC_MASKLITE_PLANES,
    )  # one list, owned by the library
except ImportError:  # pragma: no cover
    HSC_MASKLITE_PLANES = (
        "BAD",
        "SAT",
        "INTRP",
        "CR",
        "EDGE",
        "SUSPECT",
        "NO_DATA",
        "CROSSTALK",
        "UNMASKEDNAN",
    )

VARIANCE_KINDS = ("variance", "inverse_variance", "none")
PIECE_RULES = ("correlated", "independent")


@dataclass(frozen=True)
class Identity:
    """Generic identity of one target image: exposure, detector, band, and
    survey extras.
    """

    exposure_id: str
    ccd_id: str
    band: str
    extra: Mapping[str, Any] = field(default_factory=dict)

    @property
    def item_id(self) -> str:
        return f"{self.ccd_id}-v{self.exposure_id}"

    def matches(self, token: str) -> bool:
        """``--only`` tokens: ``CCD`` or ``CCD/EXPOSURE``."""
        return (
            token == self.ccd_id
            or token == f"{self.ccd_id}/{self.exposure_id}"
        )

    def columns(self) -> dict[str, Any]:
        cols = {
            "exposure_id": self.exposure_id,
            "ccd_id": self.ccd_id,
            "band": self.band,
        }
        cols.update({k: v for k, v in self.extra.items() if k not in cols})
        return cols


@dataclass
class TargetPlanes:
    """Target image planes on the reader's array module."""

    image: Any  # float, (ny, nx)
    bad: Any | None  # bool, (ny, nx); None when the survey has no mask plane
    variance: (
        Any | None
    )  # float64, (ny, nx); None when the survey has no variance
    wcs: Any
    header: Any
    mask_bits: int | None  # the resolved bad bitmask, for diagnostics
    mask: Any | None = (
        # raw integer mask plane when present (candidates are re-checked
        # against it)
        None
    )

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.image.shape[0]), int(self.image.shape[1])


class TemplateProvider:
    """Builds the template on the target grid, optionally under an affine
    refinement.

    ``build(affine)`` returns ``(image float32, bad bool | None, coverage
    count | None)``. ``affine`` is ``(A, b)`` mapping target pixel
    coordinates to refined target coordinates before the survey's own
    mapping; providers that cannot apply it set ``refinable = False``.

    A survey may supply a variance plane with its template. ``has_variance``
    says so; after a ``build``, ``variance()`` returns the template's
    variance on the target grid (float64, NaN where the template is NaN) or
    None when the survey has none, and ``resampling(x, y)`` the
    ``cuphoton.xrep.ResamplingWeights`` of the same build at target pixels,
    which the error budget's correlation term takes with the variance. A
    provider with ``has_variance`` implements both;
    :func:`check_template_provider` enforces the pairing.
    """

    refinable = False
    has_variance = False

    def build(self, affine=None):
        raise NotImplementedError

    def variance(self):
        return None

    def resampling(self, x, y):
        raise NotImplementedError


@dataclass(frozen=True)
class TemplateSource:
    """One image a template is assembled from: its pixels, the bad bits of
    its mask (or None), its WCS, and its variance plane (or None when the
    survey has none).
    """

    image: Any
    mask: Any | None
    wcs: Any
    variance: Any | None = None


def plane_map(header) -> dict[str, int]:
    """``MP_<NAME>`` cards of a mask header as ``{NAME: bit}``."""
    out = {}
    for key, value in header.items():
        if key.startswith("MP_"):
            try:
                out[key[3:].strip().upper()] = int(value)
            except (TypeError, ValueError):
                continue
    return out


def resolve_mask_bits(policy, header) -> int | None:
    """Bitmask of bad planes for ``policy``; None means "any nonzero bit"
    (strict) or no mask (none).
    """
    if policy in (None, "none"):
        return 0
    if policy == "strict":
        return None
    if policy in ("hsc-masklite", "masklite"):
        names = HSC_MASKLITE_PLANES
    elif isinstance(policy, Mapping) and "bits" in policy:
        return int(policy["bits"])
    elif isinstance(policy, Mapping) and "planes" in policy:
        names = tuple(str(n).upper() for n in policy["planes"])
    else:
        raise ValueError(
            f"unknown mask policy {policy!r}; use none, strict, hsc-"
            f"masklite, {{planes: [...]}} or {{bits: N}}"
        )
    planes = plane_map(header)
    missing = [n for n in names if n not in planes]
    if missing:
        raise ValueError(
            f"mask header lacks MP_ plane cards for {missing}; present: "
            f"{sorted(planes)}. "
            "Choose a mask policy that matches this survey (strict, or "
            "planes/bits)."
        )
    return sum(1 << planes[n] for n in names)


def bad_from_mask(mask, bits: int | None, xp):
    """Boolean bad-pixel plane from an integer mask under resolved bits."""
    if mask.dtype.kind not in "iu":
        raise TypeError(
            f"mask plane decoded as {mask.dtype}; bit tests need an integer "
            f"plane"
        )
    if bits is None:
        return mask != 0
    if bits == 0:
        return xp.zeros(mask.shape, dtype=bool)
    return (mask & bits) != 0


def variance_from(plane, kind: str, xp):
    if kind == "none" or plane is None:
        return None
    v = plane.astype(xp.float64)
    if kind == "inverse_variance":
        v = xp.where(v > 0, 1.0 / xp.where(v > 0, v, 1.0), xp.nan)
    elif kind != "variance":
        raise ValueError(
            f"variance kind must be one of {VARIANCE_KINDS}, got {kind!r}"
        )
    return v


def dilate_bool(bad, radius: int, xp):
    """Square dilation of a boolean plane by ``radius`` pixels (0 returns the
    input).
    """
    if radius <= 0:
        return bad
    from ..ops import ndimage_for

    return ndimage_for(xp).maximum_filter(
        bad, size=2 * radius + 1, mode="nearest"
    )


class SurveyAdapter:
    """Base class; subclasses declare the selection keys and implement the
    three loaders.
    """

    name = "base"
    astrometry_label = "header-wcs"
    required_keys: tuple[str, ...] = ()

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        self.params = dict(params or {})

    def validate_selection(self, sel: Mapping[str, Any], index: int) -> None:
        missing = [k for k in self.required_keys if k not in sel]
        if missing:
            raise ValueError(
                f"selection {index} lacks keys {missing}; survey "
                f"'{self.name}' needs {list(self.required_keys)}"
            )

    def identity(self, sel: Mapping[str, Any]) -> Identity:
        raise NotImplementedError

    def input_paths(self, sel: Mapping[str, Any]) -> list[str]:
        raise NotImplementedError

    def payload(self, sel: Mapping[str, Any]) -> dict[str, Any]:
        """JSON payload the worker receives; default: the selection as is."""
        return dict(sel)

    def load_target(
        self, sel: Mapping[str, Any], io: PlaneReader
    ) -> TargetPlanes:
        raise NotImplementedError

    def template(
        self,
        sel: Mapping[str, Any],
        target: TargetPlanes,
        io: PlaneReader,
        *,
        mapping_grid_step: int,
    ) -> TemplateProvider:
        """The template provider for ``sel``: :meth:`build_template`
        checked by :func:`check_template_provider`.
        """
        return check_template_provider(
            self.build_template(
                sel, target, io, mapping_grid_step=mapping_grid_step
            )
        )

    def build_template(
        self,
        sel: Mapping[str, Any],
        target: TargetPlanes,
        io: PlaneReader,
        *,
        mapping_grid_step: int,
    ) -> TemplateProvider:
        raise NotImplementedError


def check_template_provider(provider: TemplateProvider) -> TemplateProvider:
    """Check that a provider which declares ``has_variance`` implements
    ``variance()`` and ``resampling()``: the error budget's correlation
    term needs the variance plane and the resampling weights of the same
    build, and a plane without weights (or the reverse) cannot be used.
    """
    if not isinstance(provider, TemplateProvider):
        raise TypeError(
            f"template() must return a TemplateProvider, got "
            f"{type(provider).__name__}"
        )
    cls = type(provider)
    if provider.has_variance:
        missing = [
            name
            for name in ("variance", "resampling")
            if getattr(cls, name) is getattr(TemplateProvider, name)
        ]
        if missing:
            raise TypeError(
                f"{cls.__name__} declares has_variance but does not "
                f"implement {missing}; a template variance plane and the "
                "resampling weights of the same build go together"
            )
    return provider


# ---------------------------------------------------------------------------
# template providers
class GridTemplate(TemplateProvider):
    """A template already on the target grid (same shape); no refinement
    possible. ``variance`` is the survey's plane on that grid or None.
    """

    refinable = False

    def __init__(self, image, bad, xp, variance=None) -> None:
        self.image, self.bad, self.xp = image.astype(xp.float32), bad, xp
        self._variance = (
            None if variance is None else variance.astype(xp.float64)
        )
        self.has_variance = variance is not None

    def build(self, affine=None):
        return self.image, self.bad, None

    def variance(self):
        return self._variance

    def resampling(self, x, y):
        """Identity weights: the template was not resampled."""
        return identity_resampling(self.xp, x, y)


def identity_resampling(xp, x, y):
    """``ResamplingWeights`` of a template already on the target grid: one
    unit tap per position, a translation everywhere.
    """
    from cuphoton.xrep import ResamplingWeights

    x = xp.asarray(x, dtype=xp.float64).ravel()
    y = xp.asarray(y, dtype=xp.float64).ravel()
    n = int(x.shape[0])
    destination = xp.stack([xp.rint(x), xp.rint(y)], axis=1)
    return ResamplingWeights(
        weights=xp.ones((n, 1, 1), dtype=xp.float64),
        phase=xp.zeros((n, 2), dtype=xp.float64),
        is_translation=xp.ones(n, dtype=bool),
        jacobian=xp.broadcast_to(xp.eye(2), (n, 2, 2)).copy(),
        interpolation="identity",
        offsets=xp.zeros(1, dtype=xp.int64),
        source=destination,
        destination=destination.astype(xp.int64),
    )


class ReprojectedSources(TemplateProvider):
    """Mean of one or more source images reprojected through WCS onto the
    target grid.

    One source is a single reference frame on its own grid; many are coadd
    patches. The affine is composed in front of each WCS mapping, so the
    refine stage can rebuild the template.

    When every source carries a variance plane, each plane is reprojected
    with its image (the same weights, squared) and combined under
    ``pieces`` where ``n`` sources cover a pixel: ``"correlated"`` (the
    default) gives ``(mean(sigma_i))**2``, exact when the pieces' noise is
    fully correlated, as for patches cut from one coadd of the same
    exposures; ``"independent"`` gives ``sum(var_i) / n**2``, exact when
    the pieces' noise is independent, as for separate reference frames. A
    template with any source lacking a variance has none.

    The reprojection's prepared mappings are kept after a build only when
    the template has a variance plane, since only the correlation term
    reads the resampling weights; they hold one source coordinate pair per
    target pixel for every piece, hundreds of megabytes per piece on a
    four-thousand-pixel-square grid.
    """

    refinable = True

    def __init__(
        self,
        sources: Sequence[TemplateSource | tuple[Any, Any | None, Any]],
        target_wcs,
        target_shape,
        xp,
        *,
        interpolation: str = "lanczos3",
        mapping_grid_step: int = 100,
        pieces: Literal["correlated", "independent"] = "correlated",
    ) -> None:
        from cuphoton.xrep.mapping import make_wcs_pair_transform

        if pieces not in PIECE_RULES:
            raise ValueError(
                f"pieces must be one of {PIECE_RULES}, got {pieces!r}"
            )
        self.pieces = pieces
        self.sources = [
            s if isinstance(s, TemplateSource) else TemplateSource(*s)
            for s in sources
        ]
        self.base_maps = [
            make_wcs_pair_transform(target_wcs, s.wcs) for s in self.sources
        ]
        self.shape = tuple(int(v) for v in target_shape)
        self.xp = xp
        self.interpolation = interpolation
        self.step = int(mapping_grid_step)
        self.has_variance = bool(self.sources) and all(
            s.variance is not None for s in self.sources
        )
        self._variance = None
        self._built: list[tuple[Any, Any]] = []  # (spec, prepared) per piece

    def _reproject(self, source: TemplateSource, mapping):
        """One piece onto the target grid: image, mask bits, variance, and
        the spec and prepared mapping the resampling weights are read from.
        """
        from cuphoton.xrep.geometry import BBox, ReprojectionSpec
        from cuphoton.xrep.mapping import prepare_reprojection

        xp = self.xp
        ny, nx = self.shape
        spec = ReprojectionSpec(
            mapping=mapping,
            output_bbox=BBox(0, 0, nx, ny),
            interpolation=self.interpolation,
            mapping_grid_step=self.step,
        )
        image, mask = source.image, source.mask
        var = source.variance if self.has_variance else None
        prepared = prepare_reprojection(
            spec, source_shape=tuple(int(v) for v in image.shape), xp=xp
        )
        if xp is np:
            from cuphoton.xrep.backends import get_backend

            cpu = get_backend("cpu")
            r = cpu.reproject(
                np.asarray(image),
                prepared,
                spec,
                source_mask=None if mask is None else np.asarray(mask),
            )
            rvar = (
                None
                if var is None
                else cpu.reproject_variance(np.asarray(var), prepared, spec)
            )
            rimg, rmask = np.asarray(r.image), np.asarray(r.mask)
        else:
            from cuphoton.xrep.reproject import reproject_array_device

            r = reproject_array_device(
                image,
                spec,
                source_mask=mask,
                backend="cupy",
                prepared=prepared,
                source_variance=var,
            )
            rimg, rmask, rvar = r.image, r.mask, r.variance
        return rimg, (None if mask is None else rmask), rvar, spec, prepared

    def build(self, affine=None):
        """The template as the mean of the pieces with a finite pixel
        (``cnt`` counts them), its mask bits and ``cnt``; the variance
        is defined only where every contributing piece has a finite,
        nonnegative variance. An unknown contributor's noise cannot be
        inferred from the other pieces.
        """
        xp = self.xp
        ny, nx = self.shape
        acc = xp.zeros((ny, nx), xp.float64)
        cnt = xp.zeros((ny, nx), xp.int32)
        msk = xp.zeros((ny, nx), xp.int64)
        vacc = xp.zeros((ny, nx), xp.float64)
        vcnt = xp.zeros((ny, nx), xp.int32)
        any_mask = any(s.mask is not None for s in self.sources)
        self._built = []
        for source, base in zip(self.sources, self.base_maps, strict=True):
            mapping = (
                base
                if affine is None
                else (
                    lambda c, base=base, A=affine[0], b=affine[1]: base(
                        np.asarray(c) @ A.T + b
                    )
                )
            )
            img, rmask, rvar, spec, prepared = self._reproject(
                source, mapping
            )
            if self.has_variance:
                self._built.append((spec, prepared))
            good = xp.isfinite(img)
            acc += xp.where(good, img, 0.0)
            cnt += good
            if rmask is not None:
                msk |= xp.where(good, rmask.astype(xp.int64), 0)
            if rvar is not None:
                vgood = good & xp.isfinite(rvar) & (rvar >= 0)
                # correlated pieces add in sigma, independent ones in var
                vacc += xp.where(
                    vgood,
                    xp.sqrt(rvar) if self.pieces == "correlated" else rvar,
                    0.0,
                )
                vcnt += vgood
        template = xp.where(cnt > 0, acc / xp.maximum(cnt, 1), xp.nan).astype(
            xp.float32
        )
        if self.has_variance:
            n = xp.maximum(vcnt, 1)
            combined = (
                (vacc / n) ** 2
                if self.pieces == "correlated"
                else vacc / n**2
            )
            self._variance = xp.where(
                (cnt > 0) & (vcnt == cnt), combined, xp.nan
            )
        else:
            self._variance = None
        return template, (msk if any_mask else None), cnt

    def variance(self):
        return self._variance

    def resampling(self, x, y):
        """Resampling weights of the last build at target pixels ``(x, y)``:
        at each position, those of the first piece that covers it (all-zero
        weights where none does). Only a template with a variance plane
        keeps them.
        """
        import dataclasses

        from cuphoton.xrep import resampling_weights_at

        if not self.has_variance:
            raise RuntimeError(
                "resampling weights are kept only for a template with a "
                "variance plane"
            )
        if not self._built:
            raise RuntimeError("resampling weights need a build first")
        xp = self.xp
        merged = None
        for spec, prepared in self._built:
            at = resampling_weights_at(spec, x, y, prepared=prepared)
            if merged is None:
                merged = at
                continue
            covered = (merged.weights != 0).any(axis=(1, 2))
            take = ~covered & (at.weights != 0).any(axis=(1, 2))
            fields = {}
            for name in (
                "weights",
                "phase",
                "is_translation",
                "jacobian",
                "source",
                "destination",
            ):
                a, b = getattr(merged, name), getattr(at, name)
                sel = take.reshape((-1,) + (1,) * (a.ndim - 1))
                fields[name] = xp.where(sel, b, a)
            merged = dataclasses.replace(merged, **fields)
        return merged
