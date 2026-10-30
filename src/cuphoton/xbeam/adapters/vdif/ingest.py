# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Bounded VDIF extent planning for correlation windows."""

from __future__ import annotations

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal
from pathlib import Path
from typing import Any

from cuphoton.xbeam.adapters.radio.models import (
    Datastream,
    Frequency,
    Job,
    SamplingKind,
)
from cuphoton.xbeam.adapters.vdif.recording import (
    VdifSourceRange,
    VdifThreadWindow,
)
from cuphoton.xbeam.adapters.vdif.vdif import (
    VdifError,
    VdifHeader,
    _epoch_origin_seconds,
    absolute_frame_slot,
    mjd_seconds_to_vdif_seconds,
)
from cuphoton.xbeam.adapters.vdif.vdif_index import (
    VdifFrameRange,
    VdifInspection,
    VdifOrderedSearchCache,
    VdifThreadMetadata,
    VdifTimestampKey,
    inspect_vdif,
    locate_ordered_vdif_range,
    timestamp_ranges_for_slots,
)
from cuphoton.xbeam.adapters.vdif.vdif_sidecar import (
    binary_index_reference_epochs,
    query_vdif_binary_index,
    resolve_vdif_binary_index_path,
    validate_vdif_binary_index,
)
from cuphoton.xbeam.errors import ConfigurationError
from cuphoton.xbeam.profiling import nvtx_range

_MJD_UNIX_EPOCH_SECONDS = Decimal(40587 * 86400)
_DIRECT_METADATA_PROBES = 4097
_DIRECT_METADATA_MIN_PROBES = 17
_DIRECT_METADATA_PROBE_BYTES = 64 * 1024 * 1024
_MAX_ORDERED_INSPECTION_WORKERS = 16


@dataclass(frozen=True, slots=True)
class SampleInterval:
    """A half-open interval on a datastream's job-relative sample grid."""

    first_sample: int
    stop_sample: int

    def __post_init__(self) -> None:
        if self.stop_sample <= self.first_sample:
            raise ValueError("sample interval must be nonempty")

    def union(self, other: SampleInterval) -> SampleInterval:
        return SampleInterval(
            min(self.first_sample, other.first_sample),
            max(self.stop_sample, other.stop_sample),
        )


@dataclass(frozen=True, slots=True)
class VdifIngestPlan:
    """Eagerly resolved source extents and timeline bounds."""

    source_ranges: dict[int, tuple[VdifSourceRange, ...]] | None
    thread_windows: dict[int, dict[int, VdifThreadWindow]] | None
    payload_intervals: dict[tuple[int, int], SampleInterval] | None
    selected_source_bytes: int
    estimated_materialized_bytes: int | None
    estimated_decoded_bytes: int | None
    manifest: dict[str, Any]
    defer_ordered_headers: bool = False
    source_thread_windows: dict[int, dict[int, VdifThreadWindow]] | None = (
        None
    )


@dataclass(frozen=True, slots=True)
class _ThreadRequirement:
    interval: SampleInterval
    sample_rate_hz: float
    band_count: int


@dataclass(frozen=True, slots=True)
class _SourceMetadata:
    path: Path
    size_bytes: int
    mtime_ns: int
    device: int
    inode: int
    threads: dict[int, VdifThreadMetadata]
    evidence: dict[str, Any]
    inspection: VdifInspection | None = None
    reference_epochs: tuple[int, ...] = ()


_InspectionKey = tuple[Path, int, bool, Path | None]


@dataclass(slots=True)
class VdifPlanningCache:
    """Source metadata retained across bounded windows in one run."""

    inspected_sources: dict[_InspectionKey, _SourceMetadata] = field(
        default_factory=dict
    )
    ordered_searches: dict[_InspectionKey, VdifOrderedSearchCache] = field(
        default_factory=dict
    )


def _source_identity_matches(source: _SourceMetadata) -> bool:
    stat = source.path.stat()
    return (
        stat.st_size,
        stat.st_mtime_ns,
        stat.st_dev,
        stat.st_ino,
    ) == (
        source.size_bytes,
        source.mtime_ns,
        source.device,
        source.inode,
    )


def _sample_rate(datastream: Datastream, frequency: Frequency) -> float:
    if datastream.sample_rate_hz is not None:
        return datastream.sample_rate_hz
    factor = 2.0 if datastream.sampling is SamplingKind.REAL else 1.0
    return frequency.bandwidth_hz * factor


def _requirements(
    job: Job,
    intervals: Mapping[tuple[int, int], SampleInterval],
) -> dict[int, dict[int, _ThreadRequirement]]:
    frequencies = {item.identifier: item for item in job.frequencies}
    result: dict[int, dict[int, _ThreadRequirement]] = {}
    for datastream in job.datastreams:
        by_thread: dict[int, _ThreadRequirement] = {}
        for band in datastream.recorded_bands:
            interval = intervals.get((datastream.identifier, band.identifier))
            if interval is None:
                raise ConfigurationError(
                    "correlate.window",
                    "integration_count",
                    f"selected integrations contain no FFT blocks for "
                    f"datastream {datastream.identifier}, band "
                    f"{band.identifier}",
                )
            rate = _sample_rate(datastream, frequencies[band.frequency_id])
            previous = by_thread.get(band.thread_id)
            if previous is None:
                by_thread[band.thread_id] = _ThreadRequirement(
                    interval, rate, 1
                )
                continue
            if abs(previous.sample_rate_hz - rate) > max(
                1e-6, abs(rate) * 1e-12
            ):
                raise ConfigurationError(
                    f"datastream[{datastream.identifier}]",
                    "recorded_bands",
                    f"bands sharing VDIF thread {band.thread_id} imply "
                    "different sample rates",
                )
            by_thread[band.thread_id] = _ThreadRequirement(
                previous.interval.union(interval),
                previous.sample_rate_hz,
                previous.band_count + 1,
            )
        result[datastream.identifier] = by_thread
    return result


def _key_for_global_frame_slot(
    global_slot: int, frames_per_second: int
) -> VdifTimestampKey:
    absolute_seconds = Decimal(global_slot) / Decimal(frames_per_second)
    unix_seconds = absolute_seconds - _MJD_UNIX_EPOCH_SECONDS
    whole_unix_second = int(
        unix_seconds.to_integral_value(rounding=ROUND_FLOOR)
    )
    instant = datetime.fromtimestamp(whole_unix_second, tz=UTC)
    if not 2000 <= instant.year < 2032:
        raise ConfigurationError(
            "correlate.window",
            "scan_id",
            "bounded VDIF planning currently requires an epoch between "
            "2000-01-01 and 2031-12-31",
        )
    epoch_month = 1 if instant.month < 7 else 7
    reference_epoch = (instant.year - 2000) * 2 + (epoch_month == 7)
    absolute_slot = int(
        mjd_seconds_to_vdif_seconds(absolute_seconds) * frames_per_second
    )
    epoch_slot = _epoch_origin_seconds(reference_epoch, 0) * frames_per_second
    local_slot = absolute_slot - epoch_slot
    seconds_from_epoch, frame_number = divmod(local_slot, frames_per_second)
    return VdifTimestampKey(
        reference_epoch,
        seconds_from_epoch,
        frame_number,
    )


def _template_header(
    metadata: VdifThreadMetadata, key: VdifTimestampKey
) -> VdifHeader:
    return VdifHeader(
        invalid=False,
        legacy=False,
        seconds_from_epoch=key.seconds_from_epoch,
        reference_epoch=key.reference_epoch,
        frame_number=key.frame_number,
        version=metadata.version,
        log2_channels=metadata.log2_channels,
        frame_length_bytes=metadata.frame_size_bytes,
        complex_data=metadata.complex_data,
        bits_per_sample=metadata.bits_per_sample,
        thread_id=metadata.thread_id,
        station_id=metadata.station_id,
        extended_data_version=metadata.extended_data_version,
        extended_words=(
            metadata.extended_data_version << 24,
            0,
            0,
            0,
        ),
    )


def _thread_window(
    job: Job,
    requirement: _ThreadRequirement,
    metadata: VdifThreadMetadata,
) -> tuple[VdifTimestampKey, VdifTimestampKey, VdifThreadWindow]:
    rate = Decimal(str(requirement.sample_rate_hz))
    job_start = (
        Decimal(job.start_time.mjd) * Decimal(86400)
        + job.start_time.seconds_of_day
    )
    first_time = job_start + Decimal(requirement.interval.first_sample) / rate
    stop_time = job_start + Decimal(requirement.interval.stop_sample) / rate
    frames_per_second_value = (
        requirement.sample_rate_hz / metadata.samples_per_frame
    )
    frames_per_second = round(frames_per_second_value)
    if (
        frames_per_second <= 0
        or frames_per_second > (1 << 24)
        or abs(frames_per_second_value - frames_per_second) > 1e-12
    ):
        raise ConfigurationError(
            "correlate.window",
            "sample_rate_hz",
            f"thread {metadata.thread_id} does not imply an integral VDIF "
            "frame rate",
        )

    first_global_slot = (
        int(
            (first_time * frames_per_second).to_integral_value(
                rounding=ROUND_FLOOR
            )
        )
        - 1
    )
    stop_global_slot = (
        int(
            (stop_time * frames_per_second).to_integral_value(
                rounding=ROUND_CEILING
            )
        )
        + 1
    )
    start_key = _key_for_global_frame_slot(
        first_global_slot, frames_per_second
    )
    stop_key = _key_for_global_frame_slot(stop_global_slot, frames_per_second)
    start_template = _template_header(metadata, start_key)
    stop_template = _template_header(metadata, stop_key)
    return (
        start_key,
        stop_key,
        VdifThreadWindow(
            template=start_template,
            frames_per_second=frames_per_second,
            start_slot=absolute_frame_slot(start_template, frames_per_second),
            stop_slot=absolute_frame_slot(stop_template, frames_per_second),
        ),
    )


def _coalesce_ranges(
    ranges: list[VdifFrameRange], frame_size_bytes: int
) -> tuple[VdifFrameRange, ...]:
    result: list[VdifFrameRange] = []
    for item in sorted(ranges, key=lambda value: value.start_frame):
        if item.frame_size_bytes != frame_size_bytes:
            raise ConfigurationError(
                "correlate",
                "source_files",
                "VDIF range frame geometry is inconsistent",
            )
        if item.frame_count == 0:
            continue
        if result and item.start_frame <= result[-1].stop_frame:
            previous = result[-1]
            stop_frame = max(previous.stop_frame, item.stop_frame)
            result[-1] = VdifFrameRange(
                previous.start_frame,
                stop_frame,
                frame_size_bytes,
            )
        else:
            result.append(item)
    return tuple(result)


def _thread_map(
    items: tuple[VdifThreadMetadata, ...],
    *,
    entity: str,
) -> dict[int, VdifThreadMetadata]:
    result: dict[int, VdifThreadMetadata] = {}
    for item in items:
        previous = result.setdefault(item.thread_id, item)
        if previous != item:
            raise ConfigurationError(
                entity,
                "source_files",
                f"VDIF thread {item.thread_id} changes format",
            )
    return result


def _inspect_source(
    path: Path,
    frame_size_bytes: int,
    *,
    assume_ordered: bool,
    index_dir: Path | None,
    entity: str,
) -> _SourceMetadata:
    try:
        canonical = path.expanduser().resolve(strict=True)
        before = canonical.stat()
        inspection = None
        binary_index_path = resolve_vdif_binary_index_path(
            canonical, index_dir, frame_size_bytes=frame_size_bytes
        )
        if binary_index_path.is_file():
            with nvtx_range("xbeam:vdif-binary-sidecar-inspect"):
                summary = validate_vdif_binary_index(
                    canonical,
                    frame_size_bytes=frame_size_bytes,
                    index_dir=index_dir,
                )
            threads = _thread_map(summary.thread_metadata, entity=entity)
            evidence: dict[str, Any] = {
                "mode": "binary-sidecar",
                "thread_inventory_complete": True,
                "index": summary.to_dict(),
                "index_path": str(binary_index_path),
            }
            inspection = None
        elif assume_ordered:
            probe_count = min(
                _DIRECT_METADATA_PROBES,
                max(
                    _DIRECT_METADATA_MIN_PROBES,
                    _DIRECT_METADATA_PROBE_BYTES // frame_size_bytes,
                ),
            )
            with nvtx_range("xbeam:vdif-source-inspect"):
                inspection = inspect_vdif(
                    canonical,
                    frame_size_bytes=frame_size_bytes,
                    sample_count=probe_count,
                    sample_layout="edge-blocks",
                )
            if inspection.trailing_bytes:
                raise VdifError(
                    f"source has {inspection.trailing_bytes} trailing bytes"
                )
            if inspection.inconsistent_thread_ids:
                raise VdifError(
                    "sampled VDIF thread formats are inconsistent"
                )
            threads = _thread_map(inspection.sampled_threads, entity=entity)
            evidence = {
                "mode": "assume-ordered",
                "thread_inventory_complete": (
                    len(inspection.sampled_frame_indices)
                    == inspection.frame_count
                ),
                "inspection_sample_layout": "edge-blocks",
                "inspection_sample_count": probe_count,
                "inspection": inspection.to_dict(),
            }
        else:
            summary = validate_vdif_binary_index(
                canonical,
                frame_size_bytes=frame_size_bytes,
                index_dir=index_dir,
            )
            threads = _thread_map(summary.thread_metadata, entity=entity)
            evidence = {
                "mode": "binary-sidecar",
                "thread_inventory_complete": True,
                "index": summary.to_dict(),
                "index_path": str(binary_index_path),
            }
        stat = canonical.stat()
        if (
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_dev,
            stat.st_ino,
        ) != (
            before.st_size,
            before.st_mtime_ns,
            before.st_dev,
            before.st_ino,
        ):
            raise OSError(
                f"VDIF source changed during ingest planning: {canonical}"
            )
        return _SourceMetadata(
            canonical,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_dev,
            stat.st_ino,
            threads,
            evidence,
            inspection,
            tuple(
                range(
                    min(inspection.reference_epochs),
                    max(inspection.reference_epochs) + 1,
                )
            )
            if inspection is not None and inspection.reference_epochs
            else ()
            if inspection is not None
            else binary_index_reference_epochs(summary),
        )
    except OSError:
        raise
    except (TypeError, ValueError, VdifError) as exc:
        raise ConfigurationError(
            entity, "source_files", f"cannot plan {path}: {exc}"
        ) from exc


def _range_query(
    source: _SourceMetadata,
    thread_ids: tuple[int, ...],
    *,
    frame_size_bytes: int,
    assume_ordered: bool,
    index_dir: Path | None,
    entity: str,
    max_range_bytes: int,
    defer_ordered_headers: bool,
    ordered_search: VdifOrderedSearchCache | None,
    window: VdifThreadWindow,
) -> tuple[VdifFrameRange, ...]:
    ranges: list[VdifFrameRange] = []
    for start_key, stop_key in timestamp_ranges_for_slots(
        window.start_slot,
        window.stop_slot,
        window.frames_per_second,
        source.reference_epochs,
    ):
        ranges.extend(
            _raw_range_query(
                source,
                start_key,
                stop_key,
                thread_ids,
                frame_size_bytes=frame_size_bytes,
                assume_ordered=assume_ordered,
                index_dir=index_dir,
                entity=entity,
                max_range_bytes=max_range_bytes,
                defer_ordered_headers=defer_ordered_headers,
                ordered_search=ordered_search,
            )
        )
    return _coalesce_ranges(ranges, frame_size_bytes)


def _raw_range_query(
    source: _SourceMetadata,
    start_key: VdifTimestampKey,
    stop_key: VdifTimestampKey,
    thread_ids: tuple[int, ...],
    *,
    frame_size_bytes: int,
    assume_ordered: bool,
    index_dir: Path | None,
    entity: str,
    max_range_bytes: int,
    defer_ordered_headers: bool,
    ordered_search: VdifOrderedSearchCache | None,
) -> tuple[VdifFrameRange, ...]:
    try:
        if source.evidence.get("mode") == "binary-sidecar":
            with nvtx_range("xbeam:vdif-binary-sidecar-query"):
                return query_vdif_binary_index(
                    source.path,
                    start_key,
                    stop_key,
                    thread_ids=thread_ids,
                    frame_size_bytes=frame_size_bytes,
                    index_dir=index_dir,
                )
        if assume_ordered:
            with nvtx_range("xbeam:vdif-ordered-range-query"):
                return locate_ordered_vdif_range(
                    source.path,
                    start_key,
                    stop_key,
                    frame_size_bytes=frame_size_bytes,
                    thread_ids=thread_ids,
                    assume_ordered=True,
                    max_range_bytes=max_range_bytes,
                    # The streaming GPU consumer performs equivalent header
                    # validation while reading payloads. Every other consumer
                    # retains this planning-time validation pass.
                    validate_all_headers=not defer_ordered_headers,
                    search_cache=ordered_search,
                )
        raise VdifError(
            f"a valid .vdix sidecar is required for unordered input "
            f"{source.path}; run 'cuphoton-xbeam build-vdif-sidecar' or "
            "use --assume-ordered only with independent ordering evidence"
        )
    except OSError:
        raise
    except (TypeError, ValueError, VdifError) as exc:
        raise ConfigurationError(
            entity,
            "source_files",
            f"cannot resolve bounded range in {source.path}: {exc}",
        ) from exc


def _bulk_plan(
    job: Job,
    max_read_bytes: int,
    payload_intervals: Mapping[tuple[int, int], SampleInterval] | None,
) -> VdifIngestPlan:
    sources: list[dict[str, Any]] = []
    source_ranges: dict[int, tuple[VdifSourceRange, ...]] = {}
    selected_bytes = 0
    for datastream in job.datastreams:
        entity = f"datastream[{datastream.identifier}]"
        datastream_ranges: list[VdifSourceRange] = []
        for value in datastream.source_files:
            path = Path(value)
            try:
                canonical = path.expanduser().resolve(strict=True)
                before = canonical.stat()
                inspection = inspect_vdif(
                    canonical,
                    sample_count=1,
                )
            except OSError:
                raise
            except (TypeError, ValueError, VdifError) as exc:
                raise ConfigurationError(
                    entity,
                    "source_files",
                    f"cannot inspect {path}: {exc}",
                ) from exc
            after = canonical.stat()
            if (
                after.st_size,
                after.st_mtime_ns,
                after.st_dev,
                after.st_ino,
            ) != (
                before.st_size,
                before.st_mtime_ns,
                before.st_dev,
                before.st_ino,
            ):
                raise OSError(
                    f"VDIF source changed during ingest planning: {canonical}"
                )
            if inspection.frame_size_bytes != datastream.frame_size_bytes:
                raise ConfigurationError(
                    entity,
                    "frame_size_bytes",
                    f"declares {datastream.frame_size_bytes} bytes but "
                    f"{path} contains {inspection.frame_size_bytes}-byte "
                    "VDIF frames",
                )
            if inspection.trailing_bytes:
                raise ConfigurationError(
                    entity,
                    "source_files",
                    f"{path} has {inspection.trailing_bytes} trailing bytes",
                )
            selected_bytes += inspection.size_bytes
            # Bulk mode validates every header so unordered frames,
            # duplicates, format changes, and station mismatches remain
            # observable.  ``payload_intervals`` is carried separately and
            # turns these full header extents into sparse destination maps
            # before host payload materialization or GPU upload.
            datastream_ranges.append(
                VdifSourceRange(
                    inspection.path,
                    0,
                    inspection.frame_count,
                    after.st_size,
                    after.st_mtime_ns,
                    after.st_dev,
                    after.st_ino,
                )
            )
            sources.append(
                {
                    "datastream_id": datastream.identifier,
                    "path": str(inspection.path),
                    "size_bytes": inspection.size_bytes,
                    "frame_size_bytes": inspection.frame_size_bytes,
                    "fingerprint_sha256": inspection.fingerprint_sha256,
                    "mtime_ns": after.st_mtime_ns,
                    "device": after.st_dev,
                    "inode": after.st_ino,
                }
            )
        source_ranges[datastream.identifier] = tuple(datastream_ranges)
    if selected_bytes > max_read_bytes:
        raise ConfigurationError(
            "correlate",
            "max_read_bytes",
            f"unwindowed ingest requires {selected_bytes} source bytes; "
            f"limit is {max_read_bytes}; select complete integrations or "
            "raise the limit explicitly",
        )
    return VdifIngestPlan(
        source_ranges=source_ranges,
        thread_windows=None,
        payload_intervals=(
            None if payload_intervals is None else dict(payload_intervals)
        ),
        selected_source_bytes=selected_bytes,
        estimated_materialized_bytes=None,
        estimated_decoded_bytes=None,
        defer_ordered_headers=False,
        manifest={
            "schema_version": 1,
            "mode": "bulk",
            "bounded": False,
            "thread_aliases": [],
            "payload_windowed": payload_intervals is not None,
            "selected_source_bytes": selected_bytes,
            "sources": sources,
        },
    )


def plan_vdif_ingest(
    job: Job,
    *,
    intervals: Mapping[tuple[int, int], SampleInterval] | None,
    max_read_bytes: int,
    assume_ordered: bool,
    index_dir: str | Path | None,
    bulk_payload_intervals: Mapping[tuple[int, int], SampleInterval]
    | None = None,
    defer_ordered_headers: bool = False,
    planning_cache: VdifPlanningCache | None = None,
) -> VdifIngestPlan:
    """Resolve all file-backed input work before correlation output exists."""

    if max_read_bytes <= 0:
        raise ConfigurationError(
            "correlate", "max_read_bytes", "must be positive"
        )
    if defer_ordered_headers and not assume_ordered:
        raise ValueError(
            "deferred ordered-header validation requires assume_ordered=True"
        )
    if intervals is None:
        if defer_ordered_headers:
            raise ValueError(
                "deferred ordered-header validation requires bounded "
                "sample intervals"
            )
        return _bulk_plan(job, max_read_bytes, bulk_payload_intervals)

    selected_index_dir = None if index_dir is None else Path(index_dir)
    requirements = _requirements(job, intervals)
    source_ranges: dict[int, tuple[VdifSourceRange, ...]] = {}
    thread_windows: dict[int, dict[int, VdifThreadWindow]] = {}
    source_evidence: list[dict[str, Any]] = []
    thread_aliases: list[dict[str, int]] = []
    selected_bytes = 0
    materialized_bytes = 0
    decoded_bytes = 0
    effective_defer_ordered_headers = defer_ordered_headers
    single_pass_fallbacks: list[str] = []
    inspected_sources = (
        {} if planning_cache is None else planning_cache.inspected_sources
    )
    ordered_searches = (
        {} if planning_cache is None else planning_cache.ordered_searches
    )
    ordered_cache_hits: dict[_InspectionKey, bool] = {}
    resolved_ranges: dict[
        tuple[
            Path,
            int,
            int,
            tuple[int, ...],
            int,
            bool,
            bool,
            Path | None,
        ],
        tuple[VdifFrameRange, ...],
    ] = {}

    if assume_ordered:
        requests: dict[_InspectionKey, tuple[Path, int, str]] = {}
        for datastream in job.datastreams:
            entity = f"datastream[{datastream.identifier}]"
            for item in datastream.source_files:
                path = Path(item).expanduser().resolve(strict=True)
                key = (
                    path,
                    datastream.frame_size_bytes,
                    True,
                    selected_index_dir,
                )
                requests.setdefault(
                    key, (path, datastream.frame_size_bytes, entity)
                )

        missing: dict[_InspectionKey, tuple[Path, int, str]] = {}
        for key, request in requests.items():
            source = inspected_sources.get(key)
            cache_hit = source is not None
            if source is not None:
                with nvtx_range("xbeam:vdif-source-inspect-cache-check"):
                    if not _source_identity_matches(source):
                        inspected_sources.pop(key, None)
                        ordered_searches.pop(key, None)
                        source = None
                        cache_hit = False
            ordered_cache_hits[key] = cache_hit
            if source is None:
                missing[key] = request

        if missing:
            workers = min(len(missing), _MAX_ORDERED_INSPECTION_WORKERS)
            with ThreadPoolExecutor(
                max_workers=workers,
                thread_name_prefix="cuphoton-xbeam-vdif-inspect",
            ) as executor:
                futures = {
                    key: executor.submit(
                        _inspect_source,
                        path,
                        frame_size,
                        assume_ordered=True,
                        index_dir=selected_index_dir,
                        entity=entity,
                    )
                    for key, (path, frame_size, entity) in missing.items()
                }
                for key, future in futures.items():
                    inspected_sources[key] = future.result()

        for key in requests:
            source = inspected_sources[key]
            if key not in ordered_searches and source.inspection is not None:
                ordered_searches[key] = VdifOrderedSearchCache(
                    source.inspection
                )
        unordered_binary_sources = [
            str(source.path)
            for source in inspected_sources.values()
            if source.evidence.get("mode") == "binary-sidecar"
            and not source.evidence["index"]["source_ordered"]
        ]
        if effective_defer_ordered_headers and unordered_binary_sources:
            # The sidecar still supplies exact bounded physical extents, but
            # incremental publication is unsafe when physical time order is
            # arbitrary. Build destination mappings for this window first.
            effective_defer_ordered_headers = False
            single_pass_fallbacks.append(
                "unordered binary sidecar requires window-level mapping"
            )

    for datastream in job.datastreams:
        entity = f"datastream[{datastream.identifier}]"
        sources = []
        for item in datastream.source_files:
            path = Path(item).expanduser().resolve(strict=True)
            inspection_key = (
                path,
                datastream.frame_size_bytes,
                assume_ordered,
                selected_index_dir,
            )
            if assume_ordered:
                source = inspected_sources[inspection_key]
                metadata_cache_hit = ordered_cache_hits[inspection_key]
            else:
                source = inspected_sources.get(inspection_key)
                metadata_cache_hit = source is not None
                if source is not None:
                    with nvtx_range("xbeam:vdif-source-inspect-cache-check"):
                        if not _source_identity_matches(source):
                            source = None
                            metadata_cache_hit = False
                if source is None:
                    source = _inspect_source(
                        path,
                        datastream.frame_size_bytes,
                        assume_ordered=False,
                        index_dir=selected_index_dir,
                        entity=entity,
                    )
                    inspected_sources[inspection_key] = source
            sources.append((source, metadata_cache_hit))
        metadata_by_thread: dict[int, VdifThreadMetadata] = {}
        for source, metadata_cache_hit in sources:
            evidence = {
                **source.evidence,
                "metadata_cache_hit": metadata_cache_hit,
            }
            source_evidence.append(
                {
                    "datastream_id": datastream.identifier,
                    "path": str(source.path),
                    "size_bytes": source.size_bytes,
                    "mtime_ns": source.mtime_ns,
                    "device": source.device,
                    "inode": source.inode,
                    **evidence,
                }
            )
            for thread_id, observed_metadata in source.threads.items():
                previous = metadata_by_thread.setdefault(
                    thread_id, observed_metadata
                )
                if previous != observed_metadata:
                    raise ConfigurationError(
                        entity,
                        "source_files",
                        f"VDIF thread {thread_id} changes format across "
                        "source files",
                    )

        requested_threads = requirements[datastream.identifier]
        physical_threads = {
            thread_id: thread_id for thread_id in requested_threads
        }
        if (
            len(requested_threads) == 1
            and len(metadata_by_thread) == 1
            and not set(requested_threads).intersection(metadata_by_thread)
            and all(
                source.evidence["thread_inventory_complete"]
                for source, _ in sources
            )
        ):
            logical_thread = next(iter(requested_threads))
            physical_thread = next(iter(metadata_by_thread))
            physical_threads[logical_thread] = physical_thread
            thread_aliases.append(
                {
                    "datastream_id": datastream.identifier,
                    "logical_thread_id": logical_thread,
                    "physical_thread_id": physical_thread,
                }
            )

        queries: dict[int, tuple[VdifTimestampKey, VdifTimestampKey]] = {}
        planned_threads: dict[int, VdifThreadWindow] = {}
        for logical_thread_id, requirement in requested_threads.items():
            physical_thread_id = physical_threads[logical_thread_id]
            metadata = metadata_by_thread.get(physical_thread_id)
            if metadata is None:
                hint = (
                    "the bounded ordered probes did not find it; build an "
                    "index instead of using --assume-ordered"
                    if assume_ordered
                    else "the completed indexes do not contain it"
                )
                raise ConfigurationError(
                    entity,
                    "recorded_bands",
                    f"VDIF thread {logical_thread_id} is unavailable "
                    f"({hint})",
                )
            try:
                if metadata.frame_size_bytes != datastream.frame_size_bytes:
                    raise ConfigurationError(
                        entity,
                        "frame_size_bytes",
                        f"thread {physical_thread_id} uses "
                        f"{metadata.frame_size_bytes} "
                        f"bytes, expected {datastream.frame_size_bytes}",
                    )
                start_key, stop_key, planned_window = _thread_window(
                    job, requirement, metadata
                )
            except ConfigurationError as error:
                if logical_thread_id == physical_thread_id:
                    raise
                raise type(error)(
                    error.entity,
                    error.field,
                    f"{error.reason} (logical thread {logical_thread_id} "
                    f"maps to physical thread {physical_thread_id})",
                ) from error
            queries[physical_thread_id] = (start_key, stop_key)
            planned_threads[logical_thread_id] = planned_window
            slot_count = planned_window.stop_slot - planned_window.start_slot
            materialized_bytes += slot_count * (
                metadata.frame_size_bytes - 32
            )
            sample_count = slot_count * metadata.samples_per_frame
            sample_bytes = 8 if metadata.complex_data else 4
            decoded_bytes += (
                sample_count * requirement.band_count * (sample_bytes + 1)
            )
        thread_windows[datastream.identifier] = planned_threads

        planned_ranges: list[VdifSourceRange] = []
        for source, _metadata_cache_hit in sources:
            candidates: list[VdifFrameRange] = []
            query_groups: dict[
                tuple[VdifTimestampKey, VdifTimestampKey], list[int]
            ] = {}
            for thread_id, bounds in queries.items():
                query_groups.setdefault(bounds, []).append(thread_id)
            for (start_key, stop_key), thread_ids in sorted(
                query_groups.items(),
                key=lambda item: (
                    item[0][0].packed,
                    item[0][1].packed,
                    item[1],
                ),
            ):
                selected_threads = tuple(sorted(thread_ids))
                range_key = (
                    source.path,
                    start_key.packed,
                    stop_key.packed,
                    selected_threads,
                    datastream.frame_size_bytes,
                    assume_ordered,
                    effective_defer_ordered_headers,
                    selected_index_dir,
                )
                ranges = resolved_ranges.get(range_key)
                if ranges is None:
                    ranges = _range_query(
                        source,
                        selected_threads,
                        frame_size_bytes=datastream.frame_size_bytes,
                        assume_ordered=assume_ordered,
                        index_dir=selected_index_dir,
                        entity=entity,
                        max_range_bytes=max_read_bytes,
                        defer_ordered_headers=effective_defer_ordered_headers,
                        window=next(
                            window
                            for window in planned_threads.values()
                            if window.template.thread_id == thread_ids[0]
                        ),
                        ordered_search=ordered_searches.get(
                            (
                                source.path,
                                datastream.frame_size_bytes,
                                assume_ordered,
                                selected_index_dir,
                            )
                        ),
                    )
                    resolved_ranges[range_key] = ranges
                candidates.extend(ranges)
            for candidate in _coalesce_ranges(
                candidates, datastream.frame_size_bytes
            ):
                selected_bytes += candidate.byte_count
                planned_ranges.append(
                    VdifSourceRange(
                        source.path,
                        candidate.start_frame,
                        candidate.frame_count,
                        source.size_bytes,
                        source.mtime_ns,
                        source.device,
                        source.inode,
                    )
                )
        source_ranges[datastream.identifier] = tuple(planned_ranges)

    if selected_bytes > max_read_bytes:
        raise ConfigurationError(
            "correlate",
            "max_read_bytes",
            f"bounded ingest selects {selected_bytes} source bytes; limit "
            f"is {max_read_bytes}",
        )
    if materialized_bytes > max_read_bytes:
        raise ConfigurationError(
            "correlate",
            "max_read_bytes",
            f"gap-filled VDIF payloads require {materialized_bytes} bytes; "
            f"bounded materialization limit is {max_read_bytes}",
        )

    binary_sidecar_planning = bool(source_evidence) and all(
        item.get("mode") == "binary-sidecar" for item in source_evidence
    )
    if binary_sidecar_planning:
        # The sidecar supplies the exact header rows and physical locations.
        # Build a static destination map from that compact input instead of
        # enabling the raw-header streaming validator and its host progress
        # callbacks.
        effective_defer_ordered_headers = False
    estimated_range_validation_bytes = (
        selected_bytes
        if (
            assume_ordered
            and not effective_defer_ordered_headers
            and not binary_sidecar_planning
        )
        else 0
    )
    estimated_header_pass_bytes = (
        0
        if binary_sidecar_planning or effective_defer_ordered_headers
        else selected_bytes
    )
    effective_io_mode = (
        "binary-sidecar-mapped"
        if binary_sidecar_planning
        else (
            "single-pass"
            if effective_defer_ordered_headers
            else (
                "ordered-validated-three-pass"
                if assume_ordered
                else "binary-sidecar-mapped"
            )
        )
    )

    return VdifIngestPlan(
        source_ranges=source_ranges,
        thread_windows=thread_windows,
        payload_intervals=None,
        selected_source_bytes=selected_bytes,
        estimated_materialized_bytes=materialized_bytes,
        estimated_decoded_bytes=decoded_bytes,
        defer_ordered_headers=effective_defer_ordered_headers,
        manifest={
            "schema_version": 1,
            "mode": (
                "binary-sidecar"
                if binary_sidecar_planning
                else "assume-ordered"
            ),
            "bounded": True,
            "single_pass_requested": defer_ordered_headers,
            "single_pass_effective": (
                effective_defer_ordered_headers or binary_sidecar_planning
            ),
            "effective_io_mode": effective_io_mode,
            "selected_source_bytes": selected_bytes,
            "estimated_range_validation_bytes": (
                estimated_range_validation_bytes
            ),
            "estimated_header_pass_bytes": estimated_header_pass_bytes,
            "estimated_payload_pass_bytes": selected_bytes,
            "estimated_total_io_bytes": (
                estimated_range_validation_bytes
                + estimated_header_pass_bytes
                + selected_bytes
            ),
            "estimated_materialized_bytes": materialized_bytes,
            "estimated_decoded_bytes": decoded_bytes,
            "thread_aliases": thread_aliases,
            "sources": source_evidence,
            "single_pass_fallbacks": single_pass_fallbacks,
        },
    )


__all__ = [
    "SampleInterval",
    "VdifIngestPlan",
    "plan_vdif_ingest",
]
