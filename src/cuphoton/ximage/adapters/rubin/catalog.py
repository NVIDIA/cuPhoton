# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Offline validation helpers for Rubin catalog CSV files."""

from __future__ import annotations

from csv import DictReader
from pathlib import Path
from typing import Any

DEFAULT_CATALOG_PRODUCTS = ("object", "source", "forced-source")
CATALOG_PRODUCTS = {
    "dia-forced-source",
    "dia-object",
    "dia-source",
    "forced-source",
    "object",
    "source",
}
RUBIN_BANDS = {"u", "g", "r", "i", "z", "y"}
VISIT_CONSTRAINED_PRODUCTS = {
    "dia-forced-source",
    "dia-source",
    "forced-source",
    "source",
}
PRODUCT_FILE_STEMS = {
    "dia-forced-source": "dia_forced_source",
    "dia-object": "dia_object",
    "dia-source": "dia_source",
    "forced-source": "forced_source",
    "object": "object",
    "source": "source",
}
BASE_REQUIRED_COLUMNS = {
    "object": {"objectId", "coord_ra", "coord_dec"},
    "source": {
        "sourceId",
        "visit",
        "band",
        "ra",
        "dec",
        "psfFlux",
        "psfFluxErr",
        "ap12Flux",
        "ap25Flux",
        "extendedness",
        "sky",
    },
    "forced-source": {
        "objectId",
        "coord_ra",
        "coord_dec",
        "band",
        "visit",
        "detector",
        "psfFlux",
        "psfFluxErr",
        "psfDiffFlux",
        "psfDiffFluxErr",
    },
    "dia-object": {
        "diaObjectId",
        "ra",
        "dec",
        "nDiaSources",
        "radecMjdTai",
    },
    "dia-source": {
        "diaSourceId",
        "diaObjectId",
        "visit",
        "detector",
        "band",
        "ra",
        "dec",
        "coord_ra",
        "coord_dec",
        "psfFlux",
        "psfFluxErr",
        "scienceFlux",
        "scienceFluxErr",
        "reliability",
    },
    "dia-forced-source": {
        "diaObjectId",
        "coord_ra",
        "coord_dec",
        "visit",
        "detector",
        "band",
        "psfFlux",
        "psfFluxErr",
        "psfDiffFlux",
        "psfDiffFluxErr",
        "invalidPsfFlag",
    },
}


def validate_rubin_catalog_export(
    *,
    input_dir: str | Path,
    products: list[str] | tuple[str, ...] | None = None,
    visits: list[int] | tuple[int, ...] | None = None,
    band: str = "r",
) -> dict[str, Any]:
    """Validate local catalog CSV products without accessing a service."""

    product_names = normalize_catalog_products(products)
    visit_ids = normalize_visits(visits)
    safe_band = normalize_band(band)
    root = Path(input_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Catalog directory not found: {root}")
    if not root.is_dir():
        raise ValueError(f"Catalog path is not a directory: {root}")

    product_results: dict[str, Any] = {}
    for product in product_names:
        csv_path = root / f"{PRODUCT_FILE_STEMS[product]}.csv"
        product_results[product] = validate_catalog_product_csv(
            product=product,
            csv_path=csv_path,
            visits=visit_ids,
            band=safe_band,
        )

    return {
        "artifact_kind": "rubin_catalog_validation",
        "input_dir": str(root),
        "products": product_names,
        "band": safe_band,
        "expected_visits": visit_ids,
        "ok": all(item["ok"] for item in product_results.values()),
        "product_results": product_results,
    }


def validate_catalog_product_csv(
    *,
    product: str,
    csv_path: Path,
    visits: list[int],
    band: str,
) -> dict[str, Any]:
    """Validate one local catalog product against its column contract."""

    required_columns = required_columns_for_product(product, band=band)
    result: dict[str, Any] = {
        "product": product,
        "path": str(csv_path),
        "exists": csv_path.exists(),
        "ok": False,
        "row_count": 0,
        "columns": [],
        "missing_columns": sorted(required_columns),
        "visit_count": 0,
        "visits": [],
        "missing_expected_visits": (
            sorted(visits) if product in VISIT_CONSTRAINED_PRODUCTS else []
        ),
        "unexpected_visits": [],
        "band_values": [],
        "unexpected_bands": [],
        "invalid_visit_values": [],
    }
    if not csv_path.exists():
        return result

    expected_visit_set = set(visits)
    observed_visits: set[int] = set()
    invalid_visit_values: set[str] = set()
    band_values: set[str] = set()
    visit_constrained = product in VISIT_CONSTRAINED_PRODUCTS
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = DictReader(handle)
        columns = list(reader.fieldnames or [])
        column_set = set(columns)
        missing_columns = sorted(required_columns - column_set)
        visit_column = "visit" if "visit" in column_set else None
        band_column = "band" if "band" in column_set else None
        row_count = 0
        for row in reader:
            row_count += 1
            visit_value = (
                (row.get(visit_column) or "") if visit_column else ""
            )
            if visit_column is not None and (
                visit_value or visit_constrained
            ):
                try:
                    visit_id = int(visit_value)
                except ValueError:
                    invalid_visit_values.add(visit_value)
                else:
                    if visit_constrained and visit_id <= 0:
                        invalid_visit_values.add(visit_value)
                    else:
                        observed_visits.add(visit_id)
            if band_column is not None:
                band_value = row.get(band_column) or ""
                if band_value or visit_constrained:
                    band_values.add(band_value)

    unexpected_visits = (
        sorted(observed_visits - expected_visit_set)
        if expected_visit_set
        else []
    )
    missing_expected_visits = (
        sorted(expected_visit_set - observed_visits)
        if visit_column or visit_constrained
        else []
    )
    unexpected_bands = sorted(value for value in band_values if value != band)
    result.update(
        {
            "ok": not (
                missing_columns
                or invalid_visit_values
                or unexpected_visits
                or missing_expected_visits
                or unexpected_bands
            ),
            "row_count": row_count,
            "columns": columns,
            "missing_columns": missing_columns,
            "visit_count": len(observed_visits),
            "visits": sorted(observed_visits),
            "missing_expected_visits": missing_expected_visits,
            "unexpected_visits": unexpected_visits,
            "band_values": sorted(band_values),
            "unexpected_bands": unexpected_bands,
            "invalid_visit_values": sorted(invalid_visit_values),
        }
    )
    return result


def required_columns_for_product(product: str, *, band: str) -> set[str]:
    """Return the required columns for a supported local product."""

    columns = set(BASE_REQUIRED_COLUMNS[product])
    if product == "object":
        columns.update(
            {
                f"{band}_psfFlux",
                f"{band}_psfFluxErr",
                f"{band}_cModelFlux",
                f"{band}_cModelFluxErr",
                f"{band}_extendedness",
            }
        )
    elif product == "dia-object":
        columns.update(
            {
                f"{band}_psfFluxMean",
                f"{band}_psfFluxMeanErr",
                f"{band}_scienceFluxMean",
                f"{band}_scienceFluxMeanErr",
            }
        )
    return columns


def normalize_catalog_products(
    products: list[str] | tuple[str, ...] | None,
) -> list[str]:
    if products is None:
        products = DEFAULT_CATALOG_PRODUCTS
    normalized: list[str] = []
    for product in products:
        item = product.strip()
        if not item:
            continue
        if item not in CATALOG_PRODUCTS:
            allowed = ", ".join(sorted(CATALOG_PRODUCTS))
            raise ValueError(
                f"Unsupported product {item!r}; expected {allowed}"
            )
        if item not in normalized:
            normalized.append(item)
    if not normalized:
        raise ValueError("At least one Rubin catalog product is required")
    return normalized


def normalize_visits(
    visits: list[int] | tuple[int, ...] | None,
) -> list[int]:
    if visits is None:
        return []
    normalized = sorted({int(visit) for visit in visits})
    if any(visit <= 0 for visit in normalized):
        raise ValueError("visits must be positive integer visit IDs")
    return normalized


def parse_visit_text(text: str | None) -> list[int]:
    if text is None:
        return []
    visits: list[int] = []
    for token in text.replace("\n", ",").split(","):
        token = token.strip()
        if token:
            visits.append(int(token))
    return visits


def load_visit_file(path: str | Path | None) -> list[int]:
    if path is None:
        return []
    visit_path = Path(path).expanduser()
    visits: list[int] = []
    for line in visit_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        visits.append(int(line.split()[0]))
    return visits


def normalize_band(value: str) -> str:
    if value not in RUBIN_BANDS:
        allowed = ", ".join(sorted(RUBIN_BANDS))
        raise ValueError(
            f"Unsupported Rubin band {value!r}; expected {allowed}"
        )
    return value
