# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional FITS-registry integration for XScan HSC manifests."""

from __future__ import annotations

import json
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class ButlerRegistryContext:
    registry_path: Path
    collection: str | None
    datasettype: str | None
    query: str | None
    exposure_index_column: str | None
    row_count: int
    allowed_exposure_indices: list[int] | None
    rows_by_exposure: dict[int, dict[str, Any]]


@dataclass(slots=True)
class HscFitsRegistryBuildResult:
    registry_path: Path
    collections_path: Path
    summary: dict[str, Any]


_DIRECT_WARP_RE = re.compile(
    r"^deepCoadd_directWarp_(?P<instrument>[^_]+)_"
    r"(?P<tract>\d+)_(?P<patch>.+?)_"
    r"(?P<band>[A-Za-z][A-Za-z0-9]*)_"
    r"(?P<physical_filter>[^_]+)_(?P<visit>\d+)_.+$"
)
_COADD_RE = re.compile(
    r"^deepCoadd_(?P<tract>\d+)_(?P<patch>.+?)_"
    r"(?P<band>[A-Za-z][A-Za-z0-9]*)_.+$"
)
_OBJECT_TABLE_RE = re.compile(
    r"^objectTable_(?P<tract>\d+)_(?P<patch>\d+(?:[,_]\d+)*)_.+$"
)


def build_hsc_fits_registry(
    *,
    fits_root: Path,
    output_path: Path,
    hsc_npy_dir: Path | None = None,
    butler_run: str = "local-hsc-fits",
) -> HscFitsRegistryBuildResult:
    """Build a small Butler-style registry from local HSC FITS products."""
    fits_root = fits_root.expanduser().resolve()
    if not fits_root.exists():
        raise FileNotFoundError(f"HSC FITS root not found: {fits_root}")
    output_path = output_path.expanduser().resolve()

    rows: list[dict[str, Any]] = []
    warp_paths = _rglob_suffix(fits_root / "warps", ".fits")
    warp_paths_by_band: dict[str, list[Path]] = {}
    for path in warp_paths:
        parsed = _parse_hsc_path_metadata(path, fits_root=fits_root)
        band_key = str(parsed.get("band") or "unknown")
        warp_paths_by_band.setdefault(band_key, []).append(path)
    for band_key in sorted(warp_paths_by_band):
        for exposure_index, path in enumerate(warp_paths_by_band[band_key]):
            rows.append(
                _registry_row_for_path(
                    path,
                    fits_root=fits_root,
                    datasettype="visit_image",
                    butler_run=butler_run,
                    exposure_index=exposure_index,
                )
            )

    for path in _rglob_suffix(fits_root / "coadd", ".fits"):
        rows.append(
            _registry_row_for_path(
                path,
                fits_root=fits_root,
                datasettype="deepCoadd",
                butler_run=butler_run,
                exposure_index=None,
            )
        )

    for path in [
        *sorted((fits_root / "catalogs").rglob("*.parq")),
        *sorted((fits_root / "catalogs").rglob("*.parquet")),
    ]:
        rows.append(
            _registry_row_for_path(
                path,
                fits_root=fits_root,
                datasettype="objectTable",
                butler_run=butler_run,
                exposure_index=None,
            )
        )

    if not rows:
        raise ValueError(f"no FITS registry inputs found under {fits_root}")
    if hsc_npy_dir is not None:
        _validate_registry_exposure_count(
            hsc_npy_dir.expanduser().resolve(),
            warp_counts_by_band={
                band: len(paths) for band, paths in warp_paths_by_band.items()
            },
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_registry_dataframe(rows, output_path)
    collections_path = _write_default_collections_sidecar(output_path)

    summary = {
        "workflow": "data-build-hsc-registry",
        "fits_root": str(fits_root),
        "registry_path": str(output_path),
        "collections_path": str(collections_path),
        "butler_run": butler_run,
        "row_count": len(rows),
        "datasettype_counts": _value_counts(
            row["butler_datasettype"] for row in rows
        ),
        "tract_counts": _value_counts(row.get("tract") for row in rows),
        "patch_counts": _value_counts(row.get("patch") for row in rows),
        "band_counts": _value_counts(row.get("band") for row in rows),
        "fits_header_status_counts": _value_counts(
            row.get("fits_header_status") for row in rows
        ),
        "object_count_status_counts": _value_counts(
            row.get("object_count_status") for row in rows
        ),
        "warp_count": len(warp_paths),
        "warp_counts_by_band": _value_counts(
            band
            for band, paths in warp_paths_by_band.items()
            for _path in paths
        ),
        "hsc_npy_dir": (
            str(hsc_npy_dir.expanduser().resolve())
            if hsc_npy_dir is not None
            else None
        ),
    }
    return HscFitsRegistryBuildResult(
        registry_path=output_path,
        collections_path=collections_path,
        summary=summary,
    )


def resolve_butler_registry_context(
    payload: dict[str, Any],
    *,
    exposure_count: int,
) -> ButlerRegistryContext | None:
    registry_value = payload.get("butler_registry")
    if not registry_value:
        return None

    registry_path = Path(registry_value).expanduser().resolve()
    if not registry_path.exists():
        raise FileNotFoundError(f"butler_registry not found: {registry_path}")

    df = _load_registry_dataframe(registry_path)
    collection = payload.get("butler_collection")
    datasettype = payload.get("butler_datasettype")
    query = payload.get("butler_query")
    if collection:
        df = _resolve_collection(df, registry_path, str(collection))
    if datasettype is not None:
        if "butler_datasettype" not in df.columns:
            raise ValueError(
                "butler_datasettype requested but registry has no "
                "butler_datasettype column"
            )
        df = df[df["butler_datasettype"].astype("string") == str(datasettype)]
    if query:
        df = _safe_query(df, str(query), source="butler_query")
    if df.empty:
        raise ValueError("butler registry filters produced no rows")

    exposure_index_column = payload.get("butler_exposure_index_column")
    allowed: list[int] | None = None
    rows_by_exposure: dict[int, dict[str, Any]] = {}
    if exposure_index_column:
        column = str(exposure_index_column)
        if column not in df.columns:
            raise ValueError(
                "butler_exposure_index_column not present in registry: "
                f"{column}"
            )
        allowed_set: set[int] = set()
        for _, row in df.iterrows():
            value = row[column]
            if _is_missing(value):
                continue
            exposure_index = int(value)
            if exposure_index < 0 or exposure_index >= exposure_count:
                continue
            if exposure_index in rows_by_exposure:
                raise ValueError(
                    "butler_exposure_index_column produced duplicate "
                    f"exposure index {exposure_index}; add a butler_query "
                    "that selects one band or one row per exposure"
                )
            allowed_set.add(exposure_index)
            rows_by_exposure[exposure_index] = _row_metadata(row.to_dict())
        if not allowed_set:
            raise ValueError(
                "butler_exposure_index_column did not identify any valid "
                "HSC exposure indices"
            )
        allowed = sorted(allowed_set)

    return ButlerRegistryContext(
        registry_path=registry_path,
        collection=str(collection) if collection is not None else None,
        datasettype=str(datasettype) if datasettype is not None else None,
        query=str(query) if query is not None else None,
        exposure_index_column=(
            str(exposure_index_column)
            if exposure_index_column is not None
            else None
        ),
        row_count=int(len(df)),
        allowed_exposure_indices=allowed,
        rows_by_exposure=rows_by_exposure,
    )


def butler_summary(
    context: ButlerRegistryContext | None,
) -> dict[str, Any] | None:
    if context is None:
        return None
    return {
        "registry_path": str(context.registry_path),
        "collection": context.collection,
        "datasettype": context.datasettype,
        "query": context.query,
        "exposure_index_column": context.exposure_index_column,
        "row_count": context.row_count,
        "allowed_exposure_indices": context.allowed_exposure_indices,
    }


def butler_metadata_for_exposure(
    context: ButlerRegistryContext | None,
    exposure_index: int,
) -> dict[str, Any]:
    if context is None:
        return {}
    row = context.rows_by_exposure.get(exposure_index, {})
    metadata = {
        "butler_registry_path": str(context.registry_path),
        "butler_collection": context.collection,
        "butler_datasettype_filter": context.datasettype,
        "butler_query": context.query,
        "butler_exposure_index_column": context.exposure_index_column,
    }
    for key in (
        "path",
        "registry_relative_path",
        "butler_run",
        "butler_datasettype",
        "data_id",
        "instrument",
        "tract",
        "patch",
        "visit",
        "exposure",
        "exposure_index",
        "detector",
        "band",
        "filter",
        "physical_filter",
        "object_count",
    ):
        if key in row:
            metadata[f"butler_{key}"] = row[key]
    return metadata


def _load_registry_dataframe(registry_path: Path):
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pandas is required when butler_registry is configured"
        ) from exc
    suffix = registry_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(registry_path)
    if suffix == ".csv":
        return pd.read_csv(registry_path)
    if suffix in {".json", ".jsonl"}:
        return pd.read_json(registry_path, lines=suffix == ".jsonl")
    raise ValueError(
        "butler_registry must be a parquet, csv, json, or jsonl file"
    )


def _write_registry_dataframe(
    rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "pandas is required to write the HSC FITS registry"
        ) from exc
    df = pd.DataFrame(rows)
    suffix = output_path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df.to_parquet(output_path, index=False)
        return
    if suffix == ".csv":
        df.to_csv(output_path, index=False)
        return
    if suffix == ".jsonl":
        df.to_json(output_path, orient="records", lines=True)
        return
    if suffix == ".json":
        df.to_json(output_path, orient="records", indent=2)
        return
    raise ValueError(
        "HSC FITS registry output must be parquet, csv, json, or jsonl"
    )


def _write_default_collections_sidecar(registry_path: Path) -> Path:
    sidecar_path = registry_path.with_name(
        registry_path.name + ".collections.json"
    )
    if sidecar_path.exists():
        return sidecar_path
    payload = {
        "collections": [
            {
                "name": "hsc_fits_all",
                "type": "CHAINED",
                "members": [
                    "hsc_direct_warps",
                    "hsc_deep_coadds",
                    "hsc_object_tables",
                ],
            },
            {
                "name": "hsc_direct_warps",
                "type": "TAGGED",
                "filter": "butler_datasettype == 'visit_image'",
            },
            {
                "name": "hsc_deep_coadds",
                "type": "TAGGED",
                "filter": "butler_datasettype == 'deepCoadd'",
            },
            {
                "name": "hsc_object_tables",
                "type": "TAGGED",
                "filter": "butler_datasettype == 'objectTable'",
            },
        ]
    }
    sidecar_path.write_text(json.dumps(payload, indent=2) + "\n")
    return sidecar_path


def _registry_row_for_path(
    path: Path,
    *,
    fits_root: Path,
    datasettype: str,
    butler_run: str,
    exposure_index: int | None,
) -> dict[str, Any]:
    rel = path.relative_to(fits_root)
    parsed = _parse_hsc_path_metadata(path, fits_root=fits_root)
    header = (
        _read_lightweight_fits_header(path)
        if path.suffix.lower() == ".fits"
        else {}
    )
    header_status = header.pop("_status", "not_applicable")
    header_error = header.pop("_error", None)
    object_count_payload = (
        _object_table_row_count(path)
        if datasettype == "objectTable"
        else {"value": None, "status": "not_applicable", "error": None}
    )
    data_id = {
        key: parsed[key]
        for key in ("tract", "patch", "band", "visit")
        if parsed.get(key) is not None
    }
    if exposure_index is not None:
        data_id["exposure_index"] = exposure_index
    row = {
        "path": str(path),
        "registry_relative_path": str(rel),
        "butler_run": butler_run,
        "butler_datasettype": datasettype,
        "data_id": json.dumps(data_id, sort_keys=True),
        "instrument": parsed.get("instrument") or header.get("INSTRUME"),
        "tract": parsed.get("tract"),
        "patch": parsed.get("patch"),
        "band": parsed.get("band"),
        "filter": parsed.get("band"),
        "physical_filter": parsed.get("physical_filter")
        or header.get("FILTER"),
        "visit": parsed.get("visit"),
        "exposure": parsed.get("visit"),
        "exposure_index": exposure_index,
        "detector": header.get("DETECTOR"),
        "object_count": object_count_payload["value"],
        "fits_header_status": header_status,
        "fits_header_error": header_error,
        "object_count_status": object_count_payload["status"],
        "object_count_error": object_count_payload["error"],
    }
    return {key: _jsonable(value) for key, value in row.items()}


def _parse_hsc_path_metadata(
    path: Path,
    *,
    fits_root: Path,
) -> dict[str, Any]:
    rel_parts = path.relative_to(fits_root).parts
    name = path.name
    parsed: dict[str, Any] = {
        "instrument": "HSC",
        "tract": None,
        "patch": None,
        "band": None,
        "physical_filter": None,
        "visit": None,
    }

    for regex in (_DIRECT_WARP_RE, _COADD_RE, _OBJECT_TABLE_RE):
        match = regex.fullmatch(name)
        if not match:
            continue
        for key, value in match.groupdict().items():
            parsed[key] = (
                _coerce_metadata_value(value)
                if key in {"tract", "visit"}
                else value
            )
        break

    if len(rel_parts) >= 4:
        dir_tract = _coerce_metadata_value(rel_parts[1])
        dir_patch = rel_parts[2]
        if parsed["tract"] is not None and parsed["tract"] != dir_tract:
            warnings.warn(
                "HSC tract parsed from filename does not match path: "
                f"{path} filename={parsed['tract']} path={dir_tract}",
                RuntimeWarning,
                stacklevel=2,
            )
        if parsed["patch"] is not None and parsed["patch"] != dir_patch:
            warnings.warn(
                "HSC patch parsed from filename does not match path: "
                f"{path} filename={parsed['patch']} path={dir_patch}",
                RuntimeWarning,
                stacklevel=2,
            )
        parsed["tract"] = dir_tract
        parsed["patch"] = dir_patch
    if len(rel_parts) >= 5 and rel_parts[0] in {"warps", "coadd"}:
        parsed["physical_filter"] = parsed["physical_filter"] or rel_parts[3]
        parsed["band"] = parsed["band"] or _band_from_filter(rel_parts[3])
    return parsed


def _read_lightweight_fits_header(path: Path) -> dict[str, Any]:
    try:
        from astropy.io import fits
    except ModuleNotFoundError:
        return {"_status": "astropy_unavailable"}
    try:
        with fits.open(path, memmap=True) as hdul:
            header = hdul[0].header
            result: dict[str, Any] = {"_status": "read"}
            result.update(
                {
                    key: _jsonable(header[key])
                    for key in ("INSTRUME", "FILTER", "DETECTOR")
                    if key in header
                }
            )
            return result
    except _fits_read_error_types(fits) as exc:
        warnings.warn(
            f"could not read lightweight FITS header from {path}: "
            f"{type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return {
            "_status": "failed",
            "_error": f"{type(exc).__name__}: {exc}",
        }


def _object_table_row_count(path: Path) -> dict[str, Any]:
    try:
        import pyarrow.parquet as pq
    except ModuleNotFoundError:
        return {
            "value": None,
            "status": "pyarrow_unavailable",
            "error": None,
        }
    try:
        return {
            "value": int(pq.ParquetFile(path).metadata.num_rows),
            "status": "read",
            "error": None,
        }
    except _parquet_metadata_error_types() as exc:
        warnings.warn(
            f"could not read objectTable parquet metadata from {path}: "
            f"{type(exc).__name__}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return {
            "value": None,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _fits_read_error_types(fits_module) -> tuple[type[BaseException], ...]:
    error_types: tuple[type[BaseException], ...] = (
        OSError,
        ValueError,
    )
    verify_module = getattr(fits_module, "verify", None)
    for error_type in (
        getattr(verify_module, "VerifyError", None),
        getattr(fits_module, "VerifyError", None),
        getattr(fits_module, "HeaderError", None),
    ):
        if isinstance(error_type, type):
            error_types = (*error_types, error_type)
    return error_types


def _rglob_suffix(root: Path, suffix: str) -> list[Path]:
    suffix = suffix.lower()
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() == suffix
    )


def _parquet_metadata_error_types() -> tuple[type[BaseException], ...]:
    error_types: tuple[type[BaseException], ...] = (OSError, ValueError)
    try:
        from pyarrow.lib import ArrowException
    except (ImportError, AttributeError):
        return error_types
    return (*error_types, ArrowException)


def _validate_registry_exposure_count(
    hsc_npy_dir: Path,
    *,
    warp_counts_by_band: dict[str, int],
) -> None:
    images_path = hsc_npy_dir / "images.npy"
    if not images_path.exists():
        return
    images = np.load(images_path, mmap_mode="r")
    if images.ndim != 4:
        raise ValueError(
            "HSC_npy images.npy must use layout (filter, y, x, exposure)"
        )
    exposure_count = int(images.shape[-1])
    mismatches = {
        band: count
        for band, count in warp_counts_by_band.items()
        if count != exposure_count
    }
    if mismatches:
        raise ValueError(
            "HSC FITS per-band warp counts do not match HSC_npy exposure "
            f"axis: warp_counts_by_band={mismatches} "
            f"exposures={exposure_count}"
        )


def _coerce_metadata_value(value: str) -> int | str:
    try:
        return int(value)
    except ValueError:
        return value


def _band_from_filter(value: str) -> str:
    if "-" in value:
        return value.rsplit("-", 1)[-1].lower()
    return value.lower()


def _value_counts(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _resolve_collection(df, registry_path: Path, name: str):
    sidecar = _load_collections_sidecar(registry_path)
    entries = {str(entry["name"]): entry for entry in sidecar}
    if name not in entries:
        raise ValueError(f"collection {name!r} is not defined in the sidecar")
    return _resolve_collection_entry(df, entries, name, seen=set())


def _load_collections_sidecar(registry_path: Path) -> list[dict[str, Any]]:
    sidecar_path = registry_path.with_name(
        registry_path.name + ".collections.json"
    )
    if not sidecar_path.exists():
        raise FileNotFoundError(
            f"butler_collection requested but sidecar not found: "
            f"{sidecar_path}"
        )
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    entries = payload.get("collections")
    if not isinstance(entries, list):
        raise ValueError(f"{sidecar_path} is not a valid collections sidecar")
    return entries


def _resolve_collection_entry(
    df,
    entries: dict[str, dict[str, Any]],
    name: str,
    *,
    seen: set[str],
):
    if name not in entries:
        raise ValueError(f"collection {name!r} is not defined in the sidecar")
    if name in seen:
        chain = " -> ".join([*seen, name])
        raise RuntimeError(f"cycle in CHAINED collection: {chain}")
    seen = seen | {name}
    entry = entries[name]
    collection_type = str(entry.get("type", "")).upper()
    if collection_type == "RUN":
        if "butler_run" not in df.columns:
            return df.iloc[0:0]
        return df[df["butler_run"].astype("string") == str(entry["run"])]
    if collection_type == "TAGGED":
        return _safe_query(
            df,
            str(entry["filter"]),
            source=f"TAGGED collection {name!r}",
        )
    if collection_type == "CHAINED":
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "pandas is required when butler_registry is configured"
            ) from exc
        parts = [
            _resolve_collection_entry(df, entries, str(member), seen=seen)
            for member in entry.get("members", [])
        ]
        if not parts:
            return df.iloc[0:0]
        combined = pd.concat(parts, axis=0)
        if (
            "data_id" in combined.columns
            and "butler_datasettype" in combined.columns
        ):
            with_dataid = combined[combined["data_id"].notna()]
            without_dataid = combined[combined["data_id"].isna()]
            deduped = with_dataid.drop_duplicates(
                subset=["butler_datasettype", "data_id"],
                keep="first",
            )
            return pd.concat([deduped, without_dataid], axis=0)
        return combined
    raise ValueError(
        f"unsupported butler collection type for {name!r}: {collection_type}"
    )


def _row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {key: _jsonable(value) for key, value in row.items()}


def _safe_query(df, expression: str, *, source: str):
    # Registry manifests are trusted local operator inputs; this keeps pandas
    # query failures descriptive without treating manifest text as untrusted.
    try:
        return df.query(expression, engine="numexpr")
    except Exception as exc:
        raise ValueError(
            f"{source} must be a pandas query compatible with the numexpr "
            f"engine: {expression!r}"
        ) from exc


def _jsonable(value: Any) -> Any:
    if _is_missing(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        import pandas as pd
    except ModuleNotFoundError:
        pd = None
    if pd is not None:
        try:
            missing = pd.isna(value)
        except (TypeError, ValueError):
            missing = False
        if isinstance(missing, np.ndarray):
            return bool(missing.all())
        if isinstance(missing, (bool, np.bool_)):
            return bool(missing)
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return bool(np.isnan(value))
    return False
