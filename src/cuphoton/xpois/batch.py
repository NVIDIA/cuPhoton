# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Whole-image batch contracts for XPOIS."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from astropy.io import fits

from cuphoton.core.bulk import WorkItem, validate_identifier

from .data import (
    ERROR_EXTENSION_NAMES,
    FITS_SUFFIXES,
    MASK_EXTENSION_NAMES,
    VARIANCE_EXTENSION_NAMES,
    load_fit_positions,
)
from .solver_options import resolve_spatial_als_config

IMAGE_PAIR_MANIFEST_SCHEMA = "cuphoton.xpois.image-pairs/v1"
IMAGE_PAIR_MANIFEST_SCHEMA_V2 = "cuphoton.xpois.image-pairs/v2"
_SUPPORTED_MANIFEST_SCHEMAS = frozenset(
    {IMAGE_PAIR_MANIFEST_SCHEMA, IMAGE_PAIR_MANIFEST_SCHEMA_V2}
)

_ROOT_FIELDS = frozenset({"schema", "pairs"})
_PAIR_REQUIRED_FIELDS = frozenset({"id", "reference", "target"})
_PATH_FIELDS = (
    "reference",
    "target",
    "variance",
    "reference_mask",
    "target_mask",
    "fit_mask",
    "fit_positions",
)
_HDU_FIELDS = (
    "reference_hdu",
    "target_hdu",
    "variance_hdu",
    "reference_mask_hdu",
    "target_mask_hdu",
)
_PAIR_FIELDS = (
    _PAIR_REQUIRED_FIELDS | frozenset(_PATH_FIELDS) | frozenset(_HDU_FIELDS)
)
_PAIR_FIELDS_V1 = _PAIR_FIELDS - {"fit_positions"}
_SUPPORTED_BACKENDS = frozenset(
    {"auto", "cpu", "cupy", "numba-cuda", "cutile"}
)
_SUPPORTED_SOLVERS = frozenset({"constant", "spatial-als"})
_SPATIAL_ALS_BACKENDS = frozenset({"auto", "cpu", "cupy"})
_MASK_POLICIES = frozenset({"none", "strict", "hsc-masklite", "masklite"})
_INPUT_IDENTITY_KEY = "_input_identity"
_INPUT_IDENTITY_POLICY = "size-mtime-ns"
_INPUT_IDENTITY_LIMITATION = (
    "Content hashes are not computed by default; size and nanosecond mtime "
    "cannot detect content changes that preserve both values."
)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    """YAML loader that rejects duplicate mapping keys."""

    def construct_document(self, node: yaml.Node) -> Any:
        # Merge flattening mutates shared anchors after their validation.
        # Reset the validation state before constructing each document.
        self._validated_mapping_nodes: set[int] = set()
        return super().construct_document(node)


def _reject_duplicate_yaml_mapping_keys(
    loader: yaml.SafeLoader,
    node: yaml.MappingNode,
    deep: bool,
    visited: set[int] | None = None,
) -> None:
    """Reject direct duplicates, including mappings used only by merges."""

    if visited is None:
        visited = set()
    if id(node) in visited:
        return
    visited.add(id(node))

    merge_tag = "tag:yaml.org,2002:merge"
    seen: set[Any] = set()
    seen_merge = False
    for key_node, value_node in tuple(node.value):
        if key_node.tag == merge_tag:
            if seen_merge:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    "found duplicate merge key",
                    key_node.start_mark,
                )
            seen_merge = True
            merge_nodes = (
                tuple(value_node.value)
                if isinstance(value_node, yaml.SequenceNode)
                else (value_node,)
            )
            merged_keys: set[Any] = set()
            for merge_node in merge_nodes:
                if isinstance(merge_node, yaml.MappingNode):
                    _reject_duplicate_yaml_mapping_keys(
                        loader,
                        merge_node,
                        deep,
                        visited,
                    )
                    resolved = _resolved_yaml_mapping_keys(
                        loader, merge_node, deep, set()
                    )
                    overlap = merged_keys & resolved
                    if overlap:
                        key = min(overlap, key=repr)
                        raise yaml.constructor.ConstructorError(
                            "while constructing a mapping",
                            node.start_mark,
                            f"found duplicate key {key!r} across merged "
                            "mappings",
                            merge_node.start_mark,
                        )
                    merged_keys |= resolved
            continue
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in seen
        except TypeError as exc:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable key",
                key_node.start_mark,
            ) from exc
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        seen.add(key)


def _resolved_yaml_mapping_keys(
    loader: yaml.SafeLoader,
    node: yaml.MappingNode,
    deep: bool,
    visited: set[int],
) -> set[Any]:
    """Collect the keys one merge source contributes after resolution."""

    if id(node) in visited:
        return set()
    visited.add(id(node))
    merge_tag = "tag:yaml.org,2002:merge"
    keys: set[Any] = set()
    for key_node, value_node in tuple(node.value):
        if key_node.tag == merge_tag:
            merge_nodes = (
                tuple(value_node.value)
                if isinstance(value_node, yaml.SequenceNode)
                else (value_node,)
            )
            for merge_node in merge_nodes:
                if isinstance(merge_node, yaml.MappingNode):
                    keys |= _resolved_yaml_mapping_keys(
                        loader, merge_node, deep, visited
                    )
            continue
        keys.add(loader.construct_object(key_node, deep=deep))
    return keys


def _construct_unique_yaml_mapping(
    loader: _UniqueKeySafeLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[Any, Any]:
    _reject_duplicate_yaml_mapping_keys(
        loader, node, deep, loader._validated_mapping_nodes
    )
    loader.flatten_mapping(node)
    result: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_yaml_mapping,
)


@dataclass(frozen=True)
class InputFileIdentity:
    """Cheap, reproducible identity for one resolved input file."""

    path: Path
    size_bytes: int
    mtime_ns: int

    @classmethod
    def capture(cls, path: Path) -> InputFileIdentity:
        """Capture size and nanosecond mtime without reading file content."""

        resolved = path.expanduser().resolve()
        stat = resolved.stat()
        return cls(
            path=resolved,
            size_bytes=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        )

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> InputFileIdentity:
        """Restore a strict worker-side identity record."""

        values = _require_mapping(payload, "input identity")
        _reject_unknown_fields(
            values,
            frozenset({"path", "size_bytes", "mtime_ns"}),
            "input identity",
        )
        path_value = values.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError("input identity path must be a non-empty string")
        path = Path(path_value)
        if not path.is_absolute():
            raise ValueError("input identity path must be absolute")
        size_bytes = _require_non_negative_integer(
            values.get("size_bytes"), field="input identity size_bytes"
        )
        mtime_ns = _require_non_negative_integer(
            values.get("mtime_ns"), field="input identity mtime_ns"
        )
        return cls(path, size_bytes, mtime_ns)

    def to_payload(self) -> dict[str, Any]:
        """Return a JSON-compatible identity record."""

        return {
            "path": str(self.path),
            "size_bytes": self.size_bytes,
            "mtime_ns": self.mtime_ns,
        }

    def verify(self) -> None:
        """Reject an input that changed since identity capture."""

        try:
            actual = self.capture(self.path)
        except OSError as exc:
            raise RuntimeError(
                f"input changed since manifest load: {self.path}: {exc}"
            ) from exc
        if actual != self:
            raise RuntimeError(
                "input changed since manifest load: "
                f"{self.path} expected size={self.size_bytes}, "
                f"mtime_ns={self.mtime_ns}; got size={actual.size_bytes}, "
                f"mtime_ns={actual.mtime_ns}"
            )


@dataclass(frozen=True)
class ImagePairSpec:
    """One independent reference/target pair from a batch manifest."""

    item_id: str
    reference: Path
    target: Path
    reference_hdu: int | None = None
    target_hdu: int | None = None
    variance: Path | None = None
    variance_hdu: int | None = None
    reference_mask: Path | None = None
    target_mask: Path | None = None
    reference_mask_hdu: int | None = None
    target_mask_hdu: int | None = None
    fit_mask: Path | None = None
    fit_positions: Path | None = None

    def to_payload(
        self, *, include_fit_positions: bool = False
    ) -> dict[str, Any]:
        """Return the canonical JSON-compatible representation."""

        payload: dict[str, Any] = {"id": self.item_id}
        for field in _PATH_FIELDS:
            if field == "fit_positions" and not include_fit_positions:
                continue
            value = getattr(self, field)
            payload[field] = str(value) if value is not None else None
        for field in _HDU_FIELDS:
            payload[field] = getattr(self, field)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> ImagePairSpec:
        """Restore a validated pair passed to a worker."""

        values = dict(payload)
        item_id = str(values.pop("id"))
        for field in _PATH_FIELDS:
            value = values.get(field)
            values[field] = Path(value) if value is not None else None
        return cls(item_id=item_id, **values)


@dataclass(frozen=True)
class ImagePairManifest:
    """Validated image-pair manifest and its canonical identity."""

    source_path: Path
    pairs: tuple[ImagePairSpec, ...]
    input_identities: tuple[InputFileIdentity, ...]
    sha256: str
    schema: str = IMAGE_PAIR_MANIFEST_SCHEMA

    def __post_init__(self) -> None:
        if (
            not isinstance(self.schema, str)
            or self.schema not in _SUPPORTED_MANIFEST_SCHEMAS
        ):
            raise ValueError(
                "manifest schema must be one of: "
                + ", ".join(
                    repr(item) for item in sorted(_SUPPORTED_MANIFEST_SCHEMAS)
                )
            )
        if self.schema != IMAGE_PAIR_MANIFEST_SCHEMA_V2 and any(
            pair.fit_positions is not None for pair in self.pairs
        ):
            raise ValueError(
                "fit_positions require manifest schema "
                f"{IMAGE_PAIR_MANIFEST_SCHEMA_V2!r}"
            )

    def canonical_payload(self) -> dict[str, Any]:
        """Return the resolved manifest used for execution."""

        return {
            "schema": self.schema,
            "pairs": [
                pair.to_payload(
                    include_fit_positions=(
                        self.schema == IMAGE_PAIR_MANIFEST_SCHEMA_V2
                    )
                )
                for pair in self.pairs
            ],
        }

    def input_identity_payload(self) -> dict[str, Any]:
        """Return the stat-based identity contract used for this execution."""

        return {
            "schema": "cuphoton.xpois.input-identity/v1",
            "policy": _INPUT_IDENTITY_POLICY,
            "content_hashes": False,
            "limitation": _INPUT_IDENTITY_LIMITATION,
            "files": [item.to_payload() for item in self.input_identities],
        }

    def verify_input_identity(self) -> None:
        """Reject files changed since the manifest was loaded."""

        for identity in self.input_identities:
            identity.verify()

    def work_items(self) -> tuple[WorkItem, ...]:
        """Convert pairs to scheduler-neutral work items."""

        by_path = {
            identity.path: identity for identity in self.input_identities
        }
        items = []
        for pair in self.pairs:
            paths = [
                path
                for field in _PATH_FIELDS
                if (path := getattr(pair, field)) is not None
            ]
            unique_paths = sorted(set(paths), key=str)
            unique_identities = [
                by_path[path].to_payload() for path in unique_paths
            ]
            payload = pair.to_payload(
                include_fit_positions=(
                    self.schema == IMAGE_PAIR_MANIFEST_SCHEMA_V2
                )
            )
            payload[_INPUT_IDENTITY_KEY] = unique_identities
            items.append(
                WorkItem(
                    item_id=pair.item_id,
                    payload=payload,
                    weight_bytes=sum(
                        by_path[path].size_bytes for path in unique_paths
                    ),
                )
            )
        return tuple(items)


@dataclass(frozen=True)
class BatchFitOptions:
    """Global fit options shared by all image pairs."""

    kernel_shape: tuple[int, int]
    basis_sigmas: tuple[float, ...]
    basis_degrees: tuple[int, ...]
    mask_policy: str = "none"
    crop_y0: int | None = None
    crop_x0: int | None = None
    crop_height: int | None = None
    crop_width: int | None = None
    auto_stamp_mask: bool = False
    auto_stamp_size: int = 31
    auto_stamp_count: int = 5
    auto_peak_percentile: float = 99.5
    background_degree: int = 0
    flux_conserve: bool = False
    backend: str = "cupy"
    solver: str = "constant"
    spatial_degree: int | None = None
    als_iterations: int | None = None
    als_tolerance: float | None = None
    als_regularization: float | None = None

    def __post_init__(self) -> None:
        if len(self.kernel_shape) != 2 or any(
            isinstance(value, bool) or value <= 0 or value % 2 == 0
            for value in self.kernel_shape
        ):
            raise ValueError(
                "kernel_shape must contain two positive odd values"
            )
        if not self.basis_sigmas or len(self.basis_sigmas) != len(
            self.basis_degrees
        ):
            raise ValueError(
                "basis_sigmas and basis_degrees must be non-empty and aligned"
            )
        if any(
            not math.isfinite(value) or value <= 0
            for value in self.basis_sigmas
        ):
            raise ValueError("basis sigmas must be finite and positive")
        if any(value < 0 for value in self.basis_degrees):
            raise ValueError("basis degrees must be non-negative")
        if (
            isinstance(self.auto_stamp_size, bool)
            or not isinstance(self.auto_stamp_size, int)
            or self.auto_stamp_size <= 0
            or self.auto_stamp_size % 2 == 0
        ):
            raise ValueError("auto_stamp_size must be a positive odd integer")
        if self.auto_stamp_mask and self.auto_stamp_size < 3:
            raise ValueError(
                "auto_stamp_size must be at least 3 "
                "when auto_stamp_mask is enabled"
            )
        if (
            isinstance(self.auto_stamp_count, bool)
            or not isinstance(self.auto_stamp_count, int)
            or self.auto_stamp_count <= 0
        ):
            raise ValueError("auto_stamp_count must be a positive integer")
        if (
            isinstance(self.auto_peak_percentile, bool)
            or not isinstance(self.auto_peak_percentile, (int, float))
            or not math.isfinite(self.auto_peak_percentile)
            or not 0 < self.auto_peak_percentile < 100
        ):
            raise ValueError(
                "auto_peak_percentile must be a finite number "
                "strictly between 0 and 100"
            )
        if self.mask_policy not in _MASK_POLICIES:
            raise ValueError(f"unsupported mask policy: {self.mask_policy}")
        if self.backend not in _SUPPORTED_BACKENDS:
            raise ValueError(f"unsupported fit backend: {self.backend}")
        if self.solver not in _SUPPORTED_SOLVERS:
            raise ValueError(f"unsupported solver: {self.solver}")
        resolve_spatial_als_config(
            self.solver,
            background_degree=self.background_degree,
            flux_conserve=self.flux_conserve,
            spatial_degree=self.spatial_degree,
            als_iterations=self.als_iterations,
            als_tolerance=self.als_tolerance,
            als_regularization=self.als_regularization,
        )
        if (
            self.solver == "spatial-als"
            and self.backend not in _SPATIAL_ALS_BACKENDS
        ):
            raise ValueError(
                "solver 'spatial-als' supports only auto, cpu, and cupy "
                "backends"
            )
        crop = (
            self.crop_y0,
            self.crop_x0,
            self.crop_height,
            self.crop_width,
        )
        if any(value is None for value in crop) and not all(
            value is None for value in crop
        ):
            raise ValueError("specify either all crop parameters or none")
        if self.crop_y0 is not None:
            assert self.crop_x0 is not None
            assert self.crop_height is not None
            assert self.crop_width is not None
            if self.crop_y0 < 0 or self.crop_x0 < 0:
                raise ValueError("crop origins must be non-negative")
            if self.crop_height <= 0 or self.crop_width <= 0:
                raise ValueError("crop dimensions must be positive")

    def to_payload(self) -> dict[str, Any]:
        """Return JSON-compatible worker options."""

        payload = asdict(self)
        payload["kernel_shape"] = list(self.kernel_shape)
        payload["basis_sigmas"] = list(self.basis_sigmas)
        payload["basis_degrees"] = list(self.basis_degrees)
        return payload

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> BatchFitOptions:
        """Restore options passed to a distributed worker."""

        values = dict(payload)
        values["kernel_shape"] = tuple(values["kernel_shape"])
        values["basis_sigmas"] = tuple(values["basis_sigmas"])
        values["basis_degrees"] = tuple(values["basis_degrees"])
        return cls(**values)


def load_image_pair_manifest(path: Path) -> ImagePairManifest:
    """Load and strictly validate a JSON or YAML pair manifest."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"manifest does not exist: {resolved}")
    if resolved.suffix.lower() not in {".json", ".yaml", ".yml"}:
        raise ValueError("image-pair manifest must be JSON or YAML")
    try:
        if resolved.suffix.lower() == ".json":
            raw = json.loads(
                resolved.read_text(encoding="utf-8"),
                object_pairs_hook=_construct_unique_json_mapping,
            )
        else:
            raw = yaml.load(
                resolved.read_text(encoding="utf-8"),
                Loader=_UniqueKeySafeLoader,
            )
    except (json.JSONDecodeError, yaml.YAMLError, ValueError) as exc:
        raise ValueError(f"invalid image-pair manifest: {exc}") from exc
    root = _require_mapping(raw, "image-pair manifest")
    _reject_unknown_fields(root, _ROOT_FIELDS, "image-pair manifest")
    schema = root.get("schema")
    if (
        not isinstance(schema, str)
        or schema not in _SUPPORTED_MANIFEST_SCHEMAS
    ):
        raise ValueError(
            "manifest schema must be one of: "
            + ", ".join(
                repr(item) for item in sorted(_SUPPORTED_MANIFEST_SCHEMAS)
            )
        )
    assert isinstance(schema, str)
    pair_fields = (
        _PAIR_FIELDS
        if schema == IMAGE_PAIR_MANIFEST_SCHEMA_V2
        else _PAIR_FIELDS_V1
    )
    raw_pairs = root.get("pairs")
    if not isinstance(raw_pairs, list) or not raw_pairs:
        raise ValueError("manifest pairs must be a non-empty list")

    pairs: list[ImagePairSpec] = []
    seen_ids: set[str] = set()
    for index, raw_pair in enumerate(raw_pairs):
        entry = _require_mapping(raw_pair, f"pairs[{index}]")
        _reject_unknown_fields(entry, pair_fields, f"pairs[{index}]")
        missing = sorted(_PAIR_REQUIRED_FIELDS - set(entry))
        if missing:
            raise ValueError(
                f"pairs[{index}] is missing required field(s): "
                + ", ".join(missing)
            )
        item_id = validate_identifier(entry["id"], field=f"pairs[{index}].id")
        if item_id in seen_ids:
            raise ValueError(f"duplicate image-pair ID: {item_id!r}")
        seen_ids.add(item_id)
        values: dict[str, Any] = {"item_id": item_id}
        for field in _PATH_FIELDS:
            values[field] = _resolve_input_path(
                entry.get(field),
                base_dir=resolved.parent,
                field=f"pairs[{index}].{field}",
                required=field in {"reference", "target"},
            )
        for field in _HDU_FIELDS:
            values[field] = _validate_hdu(
                entry.get(field), field=f"pairs[{index}].{field}"
            )
        _validate_hdu_dependencies(values, index=index)
        pairs.append(ImagePairSpec(**values))

    canonical = {
        "schema": schema,
        "pairs": [
            pair.to_payload(
                include_fit_positions=(
                    schema == IMAGE_PAIR_MANIFEST_SCHEMA_V2
                )
            )
            for pair in pairs
        ],
    }
    input_paths = sorted(
        {
            path
            for pair in pairs
            for field in _PATH_FIELDS
            if (path := getattr(pair, field)) is not None
        },
        key=str,
    )
    input_identities = tuple(
        InputFileIdentity.capture(path) for path in input_paths
    )
    identity_payload = {
        "policy": _INPUT_IDENTITY_POLICY,
        "files": [item.to_payload() for item in input_identities],
    }
    encoded = json.dumps(
        {"manifest": canonical, "input_identity": identity_payload},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return ImagePairManifest(
        source_path=resolved,
        schema=schema,
        pairs=tuple(pairs),
        input_identities=input_identities,
        sha256=hashlib.sha256(encoded).hexdigest(),
    )


def preflight_image_pair_manifest(
    manifest: ImagePairManifest, options: BatchFitOptions
) -> None:
    """Check shapes/selectors without importing or initializing CUDA."""

    manifest.verify_input_identity()
    for pair in manifest.pairs:
        reference_shape = _probe_array_shape(
            pair.reference, pair.reference_hdu, kind="image"
        )
        target_shape = _probe_array_shape(
            pair.target, pair.target_hdu, kind="image"
        )
        _require_matching_shape(
            pair.item_id, "target", target_shape, reference_shape
        )
        if pair.variance is not None:
            variance_shape = _probe_array_shape(
                pair.variance, pair.variance_hdu, kind="variance"
            )
            _require_matching_shape(
                pair.item_id, "variance", variance_shape, reference_shape
            )
        if options.crop_y0 is not None:
            assert options.crop_x0 is not None
            assert options.crop_height is not None
            assert options.crop_width is not None
            if (
                options.crop_y0 + options.crop_height > reference_shape[0]
                or options.crop_x0 + options.crop_width > reference_shape[1]
            ):
                raise ValueError(
                    f"pair {pair.item_id!r} crop exceeds image shape "
                    f"{reference_shape}"
                )
        fit_shape = reference_shape
        if options.crop_height is not None:
            assert options.crop_width is not None
            fit_shape = (options.crop_height, options.crop_width)
        if pair.fit_mask is not None and pair.fit_positions is not None:
            raise ValueError(
                f"pair {pair.item_id!r} supplies both fit_mask and "
                "fit_positions"
            )
        if pair.fit_mask is not None:
            if pair.fit_mask.suffix.lower() != ".npy":
                raise ValueError(
                    f"pair {pair.item_id!r} fit_mask must be NPY"
                )
            mask_shape = _probe_array_shape(pair.fit_mask, None, kind="image")
            if mask_shape != reference_shape:
                _require_matching_shape(
                    pair.item_id, "fit_mask", mask_shape, fit_shape
                )
            if options.auto_stamp_mask:
                raise ValueError(
                    f"pair {pair.item_id!r} supplies fit_mask while "
                    "auto_stamp_mask is enabled"
                )
        if pair.fit_positions is not None:
            if options.solver != "spatial-als":
                raise ValueError(
                    f"pair {pair.item_id!r} fit_positions require "
                    "solver='spatial-als'"
                )
            if options.auto_stamp_mask:
                raise ValueError(
                    f"pair {pair.item_id!r} supplies fit_positions while "
                    "auto_stamp_mask is enabled"
                )
            try:
                load_fit_positions(
                    pair.fit_positions,
                    image_shape=fit_shape,
                    kernel_shape=options.kernel_shape,
                )
            except ValueError as exc:
                raise ValueError(
                    f"pair {pair.item_id!r} has invalid fit_positions: {exc}"
                ) from exc
        supplied_masks = any(
            value is not None
            for value in (
                pair.reference_mask,
                pair.target_mask,
                pair.reference_mask_hdu,
                pair.target_mask_hdu,
            )
        )
        if options.mask_policy == "none" and supplied_masks:
            raise ValueError(
                f"pair {pair.item_id!r} supplies masks with "
                "mask_policy='none'"
            )
        if options.mask_policy != "none":
            for label, image_path, mask_path, hdu in (
                (
                    "reference_mask",
                    pair.reference,
                    pair.reference_mask,
                    pair.reference_mask_hdu,
                ),
                (
                    "target_mask",
                    pair.target,
                    pair.target_mask,
                    pair.target_mask_hdu,
                ),
            ):
                if mask_path is None and image_path.suffix.lower() == ".npy":
                    raise ValueError(
                        f"pair {pair.item_id!r} {label} is required for NPY "
                        "input when mask_policy is enabled"
                    )
                path = mask_path or image_path
                shape = _probe_array_shape(path, hdu, kind="mask")
                _require_matching_shape(
                    pair.item_id, label, shape, reference_shape
                )
    manifest.verify_input_identity()


def run_image_pair_item(
    item: WorkItem,
    item_output_dir: Path,
    options: BatchFitOptions,
) -> dict[str, Any]:
    """Fit one pair after the caller has established GPU placement."""

    item_start = time.perf_counter()
    from .ois import GaussianBasisComponent
    from .workflows import run_constant_kernel_fit

    payload = dict(item.payload)
    raw_identities = payload.pop(_INPUT_IDENTITY_KEY, None)
    identities = _load_input_identities(raw_identities)
    pair = ImagePairSpec.from_payload(payload)
    expected_paths = {
        path
        for field in _PATH_FIELDS
        if (path := getattr(pair, field)) is not None
    }
    if {identity.path for identity in identities} != expected_paths:
        raise ValueError(
            "work item input identity does not match pair inputs"
        )
    _verify_input_identities(identities)
    components = [
        GaussianBasisComponent(sigma=sigma, degree=degree)
        for sigma, degree in zip(options.basis_sigmas, options.basis_degrees)
    ]
    workflow_error: Exception | None = None
    try:
        result = run_constant_kernel_fit(
            reference_path=pair.reference,
            target_path=pair.target,
            output_root=item_output_dir.parent,
            name=item_output_dir.name,
            reference_hdu=pair.reference_hdu,
            target_hdu=pair.target_hdu,
            kernel_shape=options.kernel_shape,
            components=components,
            variance_path=pair.variance,
            variance_hdu=pair.variance_hdu,
            reference_mask_path=pair.reference_mask,
            target_mask_path=pair.target_mask,
            reference_mask_hdu=pair.reference_mask_hdu,
            target_mask_hdu=pair.target_mask_hdu,
            mask_policy=options.mask_policy,
            crop_y0=options.crop_y0,
            crop_x0=options.crop_x0,
            crop_height=options.crop_height,
            crop_width=options.crop_width,
            fit_mask_path=pair.fit_mask,
            fit_positions_path=pair.fit_positions,
            auto_stamp_mask=options.auto_stamp_mask,
            auto_stamp_size=options.auto_stamp_size,
            auto_stamp_count=options.auto_stamp_count,
            auto_peak_percentile=options.auto_peak_percentile,
            background_degree=options.background_degree,
            flux_conserve=options.flux_conserve,
            backend=options.backend,
            solver=options.solver,
            spatial_degree=options.spatial_degree,
            als_iterations=options.als_iterations,
            als_tolerance=options.als_tolerance,
            als_regularization=options.als_regularization,
            workflow_name="fit_batch_item",
            run_prefix="fit-batch-item",
        )
    except Exception as exc:
        workflow_error = exc
        raise
    finally:
        try:
            _verify_input_identities(identities)
        except Exception as identity_exc:
            if workflow_error is None:
                raise
            workflow_error.add_note(
                "post-item input identity verification failed: "
                f"{type(identity_exc).__name__}: {identity_exc}"
            )
    timings = dict(result.summary.get("timings_sec", {}))
    wall = dict(result.summary.get("wall_sec", {}))
    wall["item_runner"] = time.perf_counter() - item_start
    return {
        "summary_path": str(result.run_dir / "summary.json"),
        "run_dir": str(result.run_dir),
        "solver": result.summary["solver"],
        "requested_backend": result.summary["requested_backend"],
        "backend": result.summary["backend"],
        "device": result.summary["device"],
        "runtime": result.summary.get("runtime", {}),
        "timings_sec": timings,
        "wall_sec": wall,
    }


def _load_input_identities(value: Any) -> tuple[InputFileIdentity, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError("work item input identity must be a non-empty list")
    identities = tuple(InputFileIdentity.from_payload(item) for item in value)
    paths = [identity.path for identity in identities]
    if len(set(paths)) != len(paths):
        raise ValueError("work item input identity paths must be unique")
    return identities


def _verify_input_identities(
    identities: tuple[InputFileIdentity, ...],
) -> None:
    for identity in identities:
        identity.verify()


def _construct_unique_json_mapping(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"found duplicate key {key!r}")
        result[key] = value
    return result


def _require_mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} keys must be strings")
    return value


def _reject_unknown_fields(
    payload: Mapping[str, Any], allowed: frozenset[str], field: str
) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(
            f"{field} has unknown field(s): " + ", ".join(unknown)
        )


def _resolve_input_path(
    value: Any,
    *,
    base_dir: Path,
    field: str,
    required: bool,
) -> Path | None:
    if value is None and not required:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path")
    path = Path(value.strip()).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{field} does not exist: {resolved}")
    if resolved.suffix.lower() not in FITS_SUFFIXES | {".npy"}:
        raise ValueError(f"{field} must name a FITS or NPY image: {resolved}")
    return resolved


def _validate_hdu(value: Any, *, field: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer or null")
    return value


def _require_non_negative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _validate_hdu_dependencies(
    values: Mapping[str, Any], *, index: int
) -> None:
    if values["variance_hdu"] is not None and values["variance"] is None:
        raise ValueError(f"pairs[{index}].variance_hdu requires variance")
    for hdu_field, path_field in (
        ("reference_hdu", "reference"),
        ("target_hdu", "target"),
        ("variance_hdu", "variance"),
    ):
        path = values[path_field]
        if (
            values[hdu_field] is not None
            and path is not None
            and path.suffix.lower() == ".npy"
        ):
            raise ValueError(
                f"pairs[{index}].{hdu_field} requires a FITS input"
            )
    for hdu_field, path_field, fallback_field in (
        ("reference_mask_hdu", "reference_mask", "reference"),
        ("target_mask_hdu", "target_mask", "target"),
    ):
        path = values[path_field] or values[fallback_field]
        if values[hdu_field] is not None and path.suffix.lower() == ".npy":
            raise ValueError(
                f"pairs[{index}].{hdu_field} requires a FITS input"
            )


def _probe_array_shape(
    path: Path, hdu: int | None, *, kind: str
) -> tuple[int, int]:
    if path.suffix.lower() == ".npy":
        if hdu is not None:
            raise ValueError(f"HDU selectors are not supported for {path}")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.ndim != 2:
            raise ValueError(f"{path} is not a 2D image")
        return (int(array.shape[0]), int(array.shape[1]))

    with fits.open(path, memmap=True, lazy_load_hdus=False) as hdul:
        candidates = [
            (index, _fits_image_shape(item))
            for index, item in enumerate(hdul)
        ]
        candidates = [
            (index, shape) for index, shape in candidates if shape is not None
        ]
        if hdu is not None:
            if hdu >= len(hdul):
                raise ValueError(f"HDU {hdu} is out of range for {path}")
            shape = _fits_image_shape(hdul[hdu])
            if shape is None:
                raise ValueError(f"HDU {hdu} in {path} is not a 2D image")
            return shape
        if kind == "image":
            if candidates:
                return candidates[0][1]
            raise ValueError(f"could not find a 2D image HDU in {path}")

        expected_names = (
            VARIANCE_EXTENSION_NAMES
            if kind == "variance"
            else MASK_EXTENSION_NAMES
        )
        named = []
        errors = []
        for index, shape in candidates:
            extname = (
                str(hdul[index].header.get("EXTNAME", "")).strip().upper()
            )
            if extname in expected_names:
                named.append((index, shape))
            if extname in ERROR_EXTENSION_NAMES:
                errors.append(index)
        if len(named) == 1:
            return named[0][1]
        if len(named) > 1:
            raise ValueError(
                f"multiple {kind}-like HDUs in {path}; specify one"
            )
        if kind == "variance" and len(candidates) == 1:
            if candidates[0][0] in errors:
                raise ValueError(
                    f"variance FITS {path} uses an error/sigma extension"
                )
            return candidates[0][1]
        raise ValueError(f"{kind} FITS {path} is ambiguous; specify an HDU")


def _fits_image_shape(hdu: Any) -> tuple[int, int] | None:
    if not isinstance(
        hdu,
        (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU),
    ):
        return None
    header = hdu.header
    if int(header.get("NAXIS", 0)) != 2:
        return None
    height = int(header.get("NAXIS2", 0))
    width = int(header.get("NAXIS1", 0))
    if height <= 0 or width <= 0:
        return None
    return (height, width)


def _require_matching_shape(
    item_id: str,
    label: str,
    actual: tuple[int, int],
    expected: tuple[int, int],
) -> None:
    if actual != expected:
        raise ValueError(
            f"pair {item_id!r} {label} shape {actual} does not match "
            f"expected shape {expected}"
        )


__all__ = [
    "BatchFitOptions",
    "IMAGE_PAIR_MANIFEST_SCHEMA",
    "IMAGE_PAIR_MANIFEST_SCHEMA_V2",
    "ImagePairManifest",
    "ImagePairSpec",
    "InputFileIdentity",
    "load_image_pair_manifest",
    "preflight_image_pair_manifest",
    "run_image_pair_item",
]
