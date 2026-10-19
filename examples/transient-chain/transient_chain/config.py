# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Chain configuration: one declarative schema, validated once, defaults in
one place.

``load_config`` reads the YAML, applies ``--set block.key=value`` overrides,
rejects unknown
blocks and keys, fills defaults, and checks the values the chain depends on
(odd stamp size,
reader and policy names, solver bounds). The worker then reads plain dict
values and has no
defaults of its own.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

READERS = ("auto", "xdr", "astropy")
INFERENCE_POLICIES = ("strict", "checkpoint")
STAR_FINDERS = ("gpu", "daofind")

_REQUIRED = object()


def _num(lo=None, hi=None, kind=(int, float)):
    def check(v, name):
        if isinstance(v, bool) or not isinstance(v, kind):
            raise ValueError(f"{name} must be a number, got {v!r}")
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError(f"{name} must be finite, got {v!r}")
        if lo is not None and v < lo:
            raise ValueError(f"{name} must be >= {lo}, got {v}")
        if hi is not None and v > hi:
            raise ValueError(f"{name} must be <= {hi}, got {v}")
        return v

    return check


def _int(lo=None, hi=None):
    return _num(lo, hi, kind=(int,))


def _positive_finite_num(v, name):
    _num()(v, name)
    try:
        finite = math.isfinite(v)
    except OverflowError:
        finite = False
    if not finite or v <= 0:
        raise ValueError(f"{name} must be finite and > 0, got {v}")
    return v


def _nonnegative_finite_num(v, name):
    _num()(v, name)
    if not math.isfinite(v) or v < 0:
        raise ValueError(f"{name} must be finite and >= 0, got {v}")
    return v


def _nonempty_str(v, name):
    if not isinstance(v, str) or not v.strip():
        raise ValueError(f"{name} must be a non-empty string, got {v!r}")
    return v


def _bool(v, name):
    if not isinstance(v, bool):
        raise ValueError(f"{name} must be true or false, got {v!r}")
    return v


def _choice(*options):
    def check(v, name):
        if v not in options:
            raise ValueError(
                f"{name} must be one of {list(options)}, got {v!r}"
            )
        return v

    return check


def _odd_int(v, name):
    _int(3)(v, name)
    if v % 2 == 0:
        raise ValueError(
            f"{name} must be odd (stamps are centered on a pixel), got {v}"
        )
    return v


def _num_list(v, name):
    if not isinstance(v, Sequence) or isinstance(v, str) or not v:
        raise ValueError(f"{name} must be a non-empty list of numbers")
    return [_num()(x, name) for x in v]


def _degree_list(v, name):
    if not isinstance(v, Sequence) or isinstance(v, str) or not v:
        raise ValueError(f"{name} must be a non-empty list of integers")
    return [_int(0)(x, name) for x in v]


def _opt(check):
    def wrapped(v, name):
        if v is None:
            return None
        return (
            _typed(v, check, name)
            if isinstance(check, type)
            else check(v, name)
        )

    return wrapped


def _mapping(v, name):
    if v is None:
        return {}
    if not isinstance(v, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return dict(v)


# block -> key -> (default or _REQUIRED, validator)
SCHEMA: dict[str, dict[str, tuple[Any, Any]]] = {
    "io": {
        "reader": ("auto", _choice(*READERS)),
    },
    "survey": {
        "name": ("rubin", str),
        "params": ({}, _mapping),
    },
    "align": {
        "refine": (True, _bool),
        "star_finder": ("gpu", _choice(*STAR_FINDERS)),
        "mapping_grid_step": (100, _int(1)),
        "min_affine_matches": (15, _int(3)),
        "min_shift_matches": (4, _int(1)),
        "star_fwhm_px": (_REQUIRED, _num(0.5)),
        "star_threshold_sigma": (_REQUIRED, _num(1.0)),
        "max_stars": (600, _int(4)),
        "star_box_px": (15, _odd_int),
        "centroid_iterations": (4, _int(1)),
        "match_radius_px": (_REQUIRED, _num(0.1)),
        "verify_radius_px": (_REQUIRED, _num(0.1)),
        "xcorr_window_px": (512, _int(16)),
    },
    "subtract": {
        "kernel_height": (_REQUIRED, _odd_int),
        "kernel_width": (_REQUIRED, _odd_int),
        "basis_sigmas": (_REQUIRED, _num_list),
        "basis_degrees": (_REQUIRED, _degree_list),
        "background_degree": (0, _int(0)),
        "flux_conserve": (False, _bool),
        "template_bad_dilation_px": (0, _int(0)),
        "auto_stamp_mask": (True, _bool),
        "auto_stamp_size": (31, _odd_int),
        "auto_stamp_count": (5, _int(1)),
        "auto_peak_percentile": (99.5, _num(0.0, 100.0)),
        "auto_background_filter_size": (15, _odd_int),
        "auto_peak_filter_size": (9, _odd_int),
        "auto_compact_core_size": (3, _odd_int),
    },
    "detect": {
        "threshold_sigma": (_REQUIRED, _num(0.0)),
        "min_pixels": (_REQUIRED, _int(1)),
    },
    "stamps": {
        "stamp_size": (_REQUIRED, _odd_int),
    },
    "xfit": {
        "mode": ("difference", _choice("difference")),
        "compute_dtype": ("float64", _choice("float32", "float64")),
        "solver": ({}, _mapping),
        "psf_sigma_px": (None, _opt(_positive_finite_num)),
    },
    "uq": {
        "enabled": (True, _bool),
        "kernel_fit": (True, _bool),
        "sky_astrometry_arcsec": (None, _opt(_nonnegative_finite_num)),
        "sky_astrometry_source": (None, _opt(_nonempty_str)),
    },
    "classify": {
        "inference_policy": ("strict", _choice(*INFERENCE_POLICIES)),
        "apply_noise_match": (True, _bool),
        "noise_reference_sigma": (None, _opt(_num(0.0))),
        "decision_threshold": (0.5, _num(0.0, 1.0)),
        "batch_size": (256, _int(1)),
        "checkpoint_dir": (None, _opt(str)),
    },
}

SOLVER_SCHEMA: dict[str, tuple[Any, Any]] = {
    "f_tol": (None, _opt(_num(0.0))),
    "x_tol": (None, _opt(_num(0.0))),
    "g_tol": (None, _opt(_num(0.0))),
    "max_evaluations": (None, _opt(_int(1))),
    "initial_damping": (1.0e-3, _num(0.0)),
    "damping_increase": (10.0, _num(1.0)),
    "damping_decrease": (0.3, _num(0.0, 1.0)),
}


def _validate_block(
    name: str,
    raw: Mapping[str, Any] | None,
    schema: Mapping[str, tuple[Any, Any]],
) -> dict[str, Any]:
    raw = {} if raw is None else raw
    if not isinstance(raw, Mapping):
        raise ValueError(f"config block '{name}' must be a mapping")
    unknown = sorted(set(raw) - set(schema))
    if unknown:
        raise ValueError(
            f"config block '{name}' has unknown keys {unknown}; known keys: "
            f"{sorted(schema)}"
        )
    out: dict[str, Any] = {}
    for key, (default, check) in schema.items():
        if key in raw:
            value = raw[key]
        elif default is _REQUIRED:
            raise ValueError(f"config '{name}.{key}' is required")
        else:
            value = default
        out[key] = (
            check(value, f"{name}.{key}")
            if callable(check) and not isinstance(check, type)
            else _typed(value, check, f"{name}.{key}")
        )
    return out


def _typed(value, typ, name):
    if not isinstance(value, typ):
        raise ValueError(f"{name} must be {typ.__name__}, got {value!r}")
    return value


def _parse_override(text: str) -> tuple[list[str], Any]:
    if "=" not in text:
        raise ValueError(f"override '{text}' must look like block.key=value")
    key, value = text.split("=", 1)
    path = key.strip().split(".")
    if len(path) < 2 or not all(path):
        raise ValueError(
            f"override key '{key}' must be block.key (dots for nested keys)"
        )
    return path, yaml.safe_load(value)


def apply_overrides(
    cfg: dict[str, Any], overrides: Sequence[str] | None
) -> dict[str, Any]:
    for text in overrides or ():
        path, value = _parse_override(text)
        node = cfg
        for part in path[:-1]:
            node = node.setdefault(part, {})
            if not isinstance(node, Mapping):
                raise ValueError(
                    f"override '{text}' descends into a non-mapping value"
                )
        node[path[-1]] = value
    return cfg


def validate_config(raw: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("config must be a mapping of blocks")
    unknown = sorted(set(raw) - set(SCHEMA))
    if unknown:
        raise ValueError(
            f"config has unknown blocks {unknown}; known blocks: "
            f"{sorted(SCHEMA)}"
        )
    cfg = {
        name: _validate_block(name, raw.get(name), schema)
        for name, schema in SCHEMA.items()
    }
    cfg["xfit"]["solver"] = _validate_block(
        "xfit.solver", cfg["xfit"]["solver"], SOLVER_SCHEMA
    )
    s = cfg["subtract"]
    if len(s["basis_sigmas"]) != len(s["basis_degrees"]):
        raise ValueError(
            "subtract.basis_sigmas and subtract.basis_degrees must have the "
            "same length"
        )
    u = cfg["uq"]
    if (u["sky_astrometry_arcsec"] is None) != (
        u["sky_astrometry_source"] is None
    ):
        raise ValueError(
            "uq.sky_astrometry_arcsec and uq.sky_astrometry_source are set "
            "together: the sky term is reported with its provenance or not "
            "at all"
        )
    if (
        cfg["classify"]["apply_noise_match"]
        and cfg["classify"]["noise_reference_sigma"] is None
    ):
        raise ValueError(
            "classify.noise_reference_sigma is required when "
            "classify.apply_noise_match is true; "
            "it is the robust sigma of the search stamps the classifier was "
            "trained on"
        )
    if (
        cfg["classify"]["apply_noise_match"]
        and cfg["classify"]["noise_reference_sigma"] == 0
    ):
        raise ValueError(
            "classify.noise_reference_sigma must be > 0 when "
            "classify.apply_noise_match is true"
        )
    return cfg


def load_config(
    path: Path | str, overrides: Sequence[str] | None = None
) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return validate_config(apply_overrides(dict(raw), overrides))


def config_sha256(cfg: Mapping[str, Any]) -> str:
    """Digest of the chain parameters only; the checkpoint path is provenance,
    not a parameter.
    """
    scrubbed = json.loads(json.dumps(cfg, sort_keys=True))
    scrubbed["classify"].pop("checkpoint_dir", None)
    return hashlib.sha256(
        json.dumps(scrubbed, sort_keys=True).encode()
    ).hexdigest()
