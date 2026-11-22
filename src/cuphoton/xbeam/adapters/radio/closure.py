# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Per-channel scalar closure diagnostics for host visibility records."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

import numpy as np

from cuphoton.xbeam.adapters.radio.models import VisibilityRecord
from cuphoton.xbeam.adapters.radio.output import OutputRecord
from cuphoton.xbeam.adapters.radio.swin import SwinRecord

type StationId = str | int
type Triangle = tuple[StationId, StationId, StationId]
type Quadrangle = tuple[StationId, StationId, StationId, StationId]
type RadioRecord = VisibilityRecord | OutputRecord | SwinRecord


@dataclass(frozen=True, slots=True)
class ClosureKey:
    """Fields which must agree before combining visibility channels."""

    mjd: int
    seconds_of_day: Decimal
    source_id: str | int
    configuration_id: str | int
    output_frequency_id: str | int
    polarization: str
    pulsar_bin: int = 0


@dataclass(frozen=True, slots=True)
class ClosureRecord:
    """One requested closure, with NaN values where ``valid`` is false."""

    key: ClosureKey
    kind: Literal["phase", "log_amplitude"]
    stations: tuple[StationId, ...]
    values: np.ndarray
    valid: np.ndarray


def _record_fields(
    record: RadioRecord,
) -> tuple[ClosureKey, StationId, StationId, str]:
    if isinstance(record, SwinRecord):
        if len(record.polarization) != 2:
            raise ValueError("SWIN polarization must have two characters")
        a, b = record.antenna_pair
        return (
            ClosureKey(
                record.mjd,
                Decimal(str(record.seconds)),
                record.source_id,
                record.configuration_id,
                record.frequency_id,
                record.polarization[0],
                record.pulsar_bin,
            ),
            a,
            b,
            record.polarization[1],
        )
    return (
        ClosureKey(
            record.mjd,
            record.seconds_of_day,
            record.source_id,
            record.configuration_id,
            record.output_frequency_id,
            record.polarization_a,
        ),
        record.antenna_a,
        record.antenna_b,
        record.polarization_b,
    )


def _edge(
    edges: dict[tuple[StationId, StationId], np.ndarray],
    a: StationId,
    b: StationId,
    channel_count: int,
) -> np.ndarray:
    if (a, b) in edges:
        return edges[a, b]
    if (b, a) in edges:
        return edges[b, a].conj()
    return np.full(channel_count, np.nan, dtype=np.complex128)


def compute_closures(
    records: Iterable[RadioRecord],
    *,
    triangles: Iterable[Triangle] = (),
    quadrangles: Iterable[Quadrangle] = (),
) -> list[ClosureRecord]:
    """Compute requested scalar closures without averaging or calibration.

    Triangle ``(a, b, c)`` gives ``arg(Vab Vbc Vca)`` in radians.
    Quadrangle ``(a, b, c, d)`` gives
    ``log|Vab| + log|Vcd| - log|Vac| - log|Vbd|``. Reversed baselines
    are conjugated, following xBeam's ``A * conj(B)`` convention.

    The iterable is collected by :class:`ClosureKey`, so records can arrive
    in any order. Results follow first-seen group order, with triangles
    before quadrangles in caller order. Channel counts must agree within
    each group; frequency IDs must already identify matching channel grids.
    SWIN station identifiers are zero-based indices, with float timestamps
    converted to Decimal through their string representation. No timestamp
    tolerance, station-name mapping, or channel regridding is applied.

    Autocorrelations and cross-hand products are ignored. Duplicate edges,
    including reverse copies, raise ValueError instead of being averaged.
    Missing edges, nonpositive/nonfinite exposure weights, and zero or
    nonfinite visibility amplitudes produce NaN and a false validity mask.
    Positive weights do not rescale the visibilities: xBeam reports exposure,
    not inverse noise variance. Arrays must be on the host.

    Scalar station gains cancel for simultaneous, matching co-polar
    visibilities. These diagnostics do not correct leakage or baseline
    errors, supply uncertainty estimates, or establish flux calibration.
    """

    triangles = tuple(triangles)
    quadrangles = tuple(quadrangles)
    for selections, size in ((triangles, 3), (quadrangles, 4)):
        for stations in selections:
            if len(stations) != size or len(set(stations)) != size:
                raise ValueError(
                    f"a closure requires {size} distinct station identifiers"
                )
    if not triangles and not quadrangles:
        return []

    groups: dict[
        ClosureKey, dict[tuple[StationId, StationId], np.ndarray]
    ] = {}
    for record in records:
        key, a, b, polarization_b = _record_fields(record)
        if a == b or key.polarization != polarization_b:
            continue
        if not key.seconds_of_day.is_finite():
            raise ValueError("visibility timestamp must be finite")
        spectrum = np.array(record.spectrum, dtype=np.complex128, copy=True)
        if spectrum.ndim != 1 or spectrum.size == 0:
            raise ValueError("visibility spectrum must be nonempty and 1D")
        edges = groups.setdefault(key, {})
        if (a, b) in edges or (b, a) in edges:
            raise ValueError(f"duplicate baseline {a!r}, {b!r} for {key!r}")
        if edges and spectrum.shape != next(iter(edges.values())).shape:
            raise ValueError(f"inconsistent channel counts for {key!r}")
        if not np.isfinite(record.weight) or record.weight <= 0:
            spectrum = np.full(spectrum.shape, np.nan, dtype=np.complex128)
        edges[a, b] = spectrum

    results = []
    for key, edges in groups.items():
        channel_count = next(iter(edges.values())).size

        for stations in triangles:
            a, b, c = stations
            spectra = np.array(
                [
                    _edge(edges, x, y, channel_count)
                    for x, y in ((a, b), (b, c), (c, a))
                ]
            )
            amplitudes = np.abs(spectra)
            valid = np.all(np.isfinite(amplitudes) & (amplitudes > 0), axis=0)
            # Sum angles instead of forming a potentially overflowing product.
            angle = np.angle(spectra).sum(axis=0)
            values = np.arctan2(np.sin(angle), np.cos(angle))
            results.append(
                ClosureRecord(
                    key,
                    "phase",
                    stations,
                    np.where(valid, values, np.nan),
                    valid,
                )
            )
        for stations in quadrangles:
            a, b, c, d = stations
            amplitudes = np.abs(
                [
                    _edge(edges, x, y, channel_count)
                    for x, y in ((a, b), (c, d), (a, c), (b, d))
                ]
            )
            valid = np.all(np.isfinite(amplitudes) & (amplitudes > 0), axis=0)
            # Invalid channels remain NaN without divide-by-zero warnings.
            logs = np.full(amplitudes.shape, np.nan)
            np.log(amplitudes, out=logs, where=valid[None, :])
            values = logs[0] + logs[1] - logs[2] - logs[3]
            results.append(
                ClosureRecord(key, "log_amplitude", stations, values, valid)
            )
    return results
