# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Strict adapter for colon-delimited interoperability metadata."""

from __future__ import annotations

import re
import warnings
from decimal import Decimal
from pathlib import Path

from cuphoton.xbeam.adapters.radio.models import (
    Baseline,
    BaselineProduct,
    BufferedReaderGeometry,
    ClockPolynomial,
    Datastream,
    DelayCoordinateModel,
    Frequency,
    Job,
    MJDTime,
    OutputConfiguration,
    OutputFormat,
    RecordedBand,
    SamplingKind,
    Scan,
    Sideband,
    Station,
    UnsupportedFeatures,
    ZoomBand,
)
from cuphoton.xbeam.errors import JobLoadError, UnsupportedFeatureError
from cuphoton.xbeam.job_text import (
    JobText,
    parse_calc_file,
    parse_im_file,
    parse_input_file,
)
from cuphoton.xbeam.validation import validate_job


def _one(document: JobText, key: str) -> str:
    try:
        return document.get_unique(key)
    except (KeyError, ValueError) as exc:
        raise JobLoadError(
            f"{document.source}: required field {key!r} is missing or "
            "repeated"
        ) from exc


def _buffered_reader_geometry(
    document: JobText,
) -> BufferedReaderGeometry | None:
    fields = {
        "buffer_factor": "DATA BUFFER FACTOR",
        "data_segments": "NUM DATA SEGMENTS",
        "buffered_ffts": "NUM BUFFERED FFTS",
    }
    if not any(document.get_all(key) for key in fields.values()):
        return None
    values = {}
    for name, key in fields.items():
        raw = _one(document, key)
        try:
            value = int(raw)
        except ValueError as exc:
            raise JobLoadError(
                f"{document.source}: {key} must be an integer"
            ) from exc
        if value <= 0:
            raise JobLoadError(f"{document.source}: {key} must be positive")
        values[name] = value
    return BufferedReaderGeometry(**values)


def _nth(document: JobText, key: str, index: int) -> str:
    values = document.get_all(key)
    try:
        return values[index]
    except IndexError as exc:
        raise JobLoadError(
            f"{document.source}: field {key!r} occurrence {index} is missing"
        ) from exc


def _integer(document: JobText, key: str) -> int:
    try:
        return int(_one(document, key))
    except ValueError as exc:
        raise JobLoadError(
            f"{document.source}: field {key!r} must be an integer"
        ) from exc


def _boolean(value: str) -> bool:
    normalized = value.strip().upper()
    if normalized in {"TRUE", "1", "YES"}:
        return True
    if normalized in {"FALSE", "0", "NO"}:
        return False
    raise JobLoadError(f"invalid boolean value {value!r}")


def _clock_epoch(mjd: Decimal) -> MJDTime:
    day = int(mjd)
    return MJDTime(day, (mjd - day) * Decimal(86400))


def _coefficients(
    document: JobText,
    key: str,
    *,
    scale: float = 1.0,
    occurrence: int | None = None,
) -> tuple[float, ...]:
    raw = (
        _one(document, key)
        if occurrence is None
        else _nth(document, key, occurrence)
    )
    try:
        return tuple(float(item) * scale for item in raw.split())
    except ValueError as exc:
        raise JobLoadError(
            f"{document.source}: field {key!r} contains a non-numeric "
            "coefficient"
        ) from exc


def _datastream_sections(
    document: JobText, count: int
) -> tuple[JobText, ...]:
    starts = [
        index
        for index, entry in enumerate(document.entries)
        if entry.key == "TELESCOPE INDEX"
    ]
    if len(starts) != count:
        raise JobLoadError(
            f"{document.source}: expected {count} datastream sections, "
            f"found {len(starts)}"
        )
    sections = []
    for position, start in enumerate(starts):
        end = (
            starts[position + 1]
            if position + 1 < count
            else len(document.entries)
        )
        sections.append(JobText(document.entries[start:end], document.source))
    return tuple(sections)


def _baseline_sections(document: JobText, count: int) -> tuple[JobText, ...]:
    starts = []
    for baseline_index in range(count):
        key = f"D/STREAM A INDEX {baseline_index}"
        starts.append(
            next(
                (
                    index
                    for index, entry in enumerate(document.entries)
                    if entry.key == key
                ),
                -1,
            )
        )
    if any(index < 0 for index in starts) or starts != sorted(starts):
        raise JobLoadError(
            f"{document.source}: baseline sections are missing or out of "
            "order"
        )
    sections = []
    for position, start in enumerate(starts):
        end = (
            starts[position + 1]
            if position + 1 < count
            else len(document.entries)
        )
        sections.append(JobText(document.entries[start:end], document.source))
    return tuple(sections)


def _im_scan_sections(document: JobText, count: int) -> tuple[JobText, ...]:
    starts = []
    for scan_index in range(count):
        key = f"SCAN {scan_index} POINTING SRC"
        starts.append(
            next(
                (
                    index
                    for index, entry in enumerate(document.entries)
                    if entry.key == key
                ),
                -1,
            )
        )
    if any(index < 0 for index in starts) or starts != sorted(starts):
        raise JobLoadError(
            f"{document.source}: scan model sections are missing or out of "
            "order"
        )
    return tuple(
        JobText(
            document.entries[
                start : (
                    starts[position + 1]
                    if position + 1 < count
                    else len(document.entries)
                )
            ],
            document.source,
        )
        for position, start in enumerate(starts)
    )


def _sampling(value: str, entity: str) -> SamplingKind:
    normalized = value.strip().upper()
    if normalized == "REAL":
        return SamplingKind.REAL
    if normalized == "COMPLEX":
        return SamplingKind.COMPLEX
    if normalized == "COMPLEX_DSB":
        return SamplingKind.COMPLEX_DSB
    raise UnsupportedFeatureError(
        entity, "DATA SAMPLING", f"unsupported sampling kind {value!r}"
    )


def _format_threads(
    value: str, band_count: int, frame_size_bytes: int, entity: str
) -> tuple[str, tuple[int, ...]]:
    normalized = value.strip().upper()
    if normalized == "VDIF":
        # In the supplied text profile, ordinary 8032-byte VDIF frames pack
        # multiple logical bands as channels of thread 0.  The 4032-byte
        # thread-multiplexed profile carries one band per default-numbered
        # thread.  Non-default thread numbers use INTERLACEDVDIF below.
        if frame_size_bytes == 4032 and band_count > 1:
            return "VDIF", tuple(range(band_count))
        return "VDIF", (0,) * band_count
    match = re.fullmatch(
        r"INTERLACEDVDIF/([0-9]+(?::[0-9]+)*)/[0-9]+/[0-9]+", normalized
    )
    if not match:
        raise UnsupportedFeatureError(
            entity, "DATA FORMAT", f"unsupported data format {value!r}"
        )
    thread_ids = tuple(int(item) for item in match.group(1).split(":"))
    if len(thread_ids) != band_count:
        raise JobLoadError(
            f"{entity}: DATA FORMAT declares {len(thread_ids)} threads for "
            f"{band_count} recorded bands"
        )
    return "VDIF", thread_ids


def _sky_low(frequency: Frequency) -> float:
    if frequency.sideband is Sideband.USB:
        return frequency.band_edge_hz
    return frequency.band_edge_hz - frequency.bandwidth_hz


def _zoom_parent_and_start(
    output: Frequency,
    recorded_bands: tuple[RecordedBand, ...],
    frequencies: tuple[Frequency, ...],
    polarization: str,
    entity: str,
) -> tuple[int, int]:
    output_low = _sky_low(output)
    output_high = output_low + output.bandwidth_hz
    candidates = []
    for band in recorded_bands:
        parent = frequencies[band.frequency_id]
        parent_low = _sky_low(parent)
        parent_high = parent_low + parent.bandwidth_hz
        tolerance = max(parent.bandwidth_hz, output.bandwidth_hz) * 1e-12
        if (
            band.polarization == polarization
            and output_low >= parent_low - tolerance
            and output_high <= parent_high + tolerance
        ):
            channel_width = parent.bandwidth_hz / parent.channel_count
            raw_start = (output_low - parent_low) / channel_width
            start = round(raw_start)
            if abs(raw_start - start) <= 1e-7:
                candidates.append((band.identifier, start))
    if len(candidates) != 1:
        raise JobLoadError(
            f"{entity}: zoom frequency must map channel-aligned into exactly "
            "one recorded band"
        )
    return candidates[0]


def _band_output_frequency(
    datastream: Datastream, band_id: int, entity: str
) -> int:
    matches = [
        band.frequency_id
        for band in datastream.recorded_bands
        if band.identifier == band_id
    ] + [
        band.output_frequency_id
        for band in datastream.zoom_bands
        if band.identifier == band_id
    ]
    if len(matches) != 1:
        raise JobLoadError(
            f"{entity}: band {band_id} does not identify exactly one "
            "recorded or zoom frequency"
        )
    return matches[0]


def load_interop_job(path: str | Path) -> Job:
    """Load the strict colon-delimited interoperability metadata profile."""

    input_path = Path(path).resolve()
    input_doc = parse_input_file(input_path)
    calc_path = (
        input_path.parent / _one(input_doc, "CALC FILENAME")
    ).resolve()
    calc_doc = parse_calc_file(calc_path)
    im_path = (calc_path.parent / _one(calc_doc, "IM FILENAME")).resolve()
    im_doc = parse_im_file(im_path)

    if _one(input_doc, "OUTPUT FORMAT").upper() != "SWIN":
        raise UnsupportedFeatureError(
            "job", "OUTPUT FORMAT", "interop profile requires SWIN"
        )
    scan_count = _integer(calc_doc, "NUM SCANS")
    if _integer(im_doc, "NUM SCANS") != scan_count:
        raise JobLoadError(
            f"{im_doc.source}: NUM SCANS does not match the calculation "
            "metadata"
        )
    for scan_index in range(scan_count):
        if int(_one(calc_doc, f"SCAN {scan_index} NUM PHS CTRS")) != 1:
            raise UnsupportedFeatureError(
                f"scan[{scan_index}]",
                "phase_centres",
                "multiple phase centres are unsupported",
            )

    start_mjd = _integer(input_doc, "START MJD")
    start_seconds = Decimal(_one(input_doc, "START SECONDS"))
    station_count = _integer(input_doc, "TELESCOPE ENTRIES")
    stations = []
    for index in range(station_count):
        order = int(_one(input_doc, f"CLOCK POLY ORDER {index}"))
        coefficients = tuple(
            float(_one(input_doc, f"CLOCK COEFF {index}/{term}")) * 1e-6
            for term in range(order + 1)
        )
        stations.append(
            Station(
                index,
                _one(input_doc, f"TELESCOPE NAME {index}"),
                ClockPolynomial(
                    _clock_epoch(
                        Decimal(_one(input_doc, f"CLOCK REF MJD {index}"))
                    ),
                    coefficients,
                ),
            )
        )

    frequency_count = _integer(input_doc, "FREQ ENTRIES")
    frequencies = []
    for index in range(frequency_count):
        sideband_value = _one(input_doc, f"SIDEBAND {index}").upper()
        if sideband_value not in {"U", "L"}:
            raise JobLoadError(
                f"{input_doc.source}: SIDEBAND {index} must be U or L"
            )
        frequencies.append(
            Frequency(
                identifier=index,
                band_edge_hz=float(_one(input_doc, f"FREQ (MHZ) {index}"))
                * 1e6,
                bandwidth_hz=float(_one(input_doc, f"BW (MHZ) {index}"))
                * 1e6,
                sideband=Sideband.USB
                if sideband_value == "U"
                else Sideband.LSB,
                channel_count=int(_one(input_doc, f"NUM CHANNELS {index}")),
                channels_to_average=int(
                    _one(input_doc, f"CHANS TO AVG {index}")
                ),
                oversampling=int(_one(input_doc, f"OVERSAMPLE FAC. {index}")),
                decimation=int(_one(input_doc, f"DECIMATION FAC. {index}")),
            )
        )

    frequency_tuple = tuple(frequencies)
    datastream_count = _integer(input_doc, "DATASTREAM ENTRIES")
    datastreams = []
    for index, section in enumerate(
        _datastream_sections(input_doc, datastream_count)
    ):
        entity = f"datastream[{index}]"
        recorded_frequency_count = int(_one(section, "NUM RECORDED FREQS"))
        if recorded_frequency_count < 1:
            raise JobLoadError(
                f"{section.source}: {entity} NUM RECORDED FREQS "
                "must be positive"
            )
        recorded_frequency_ids = tuple(
            int(_one(section, f"REC FREQ INDEX {slot}"))
            for slot in range(recorded_frequency_count)
        )
        for slot, frequency_id in enumerate(recorded_frequency_ids):
            if not 0 <= frequency_id < frequency_count:
                raise JobLoadError(
                    f"{section.source}: {entity} REC FREQ INDEX {slot} "
                    "is outside the frequency table"
                )
        recorded_band_count = sum(
            int(_one(section, f"NUM REC POLS {slot}"))
            for slot in range(recorded_frequency_count)
        )
        frame_size_bytes = int(_one(section, "DATA FRAME SIZE"))
        data_format, thread_ids = _format_threads(
            _one(section, "DATA FORMAT"),
            recorded_band_count,
            frame_size_bytes,
            entity,
        )
        recorded_bands = []
        for band_index in range(recorded_band_count):
            frequency_slot = int(
                _one(section, f"REC BAND {band_index} INDEX")
            )
            if not 0 <= frequency_slot < recorded_frequency_count:
                raise JobLoadError(
                    f"{section.source}: REC BAND {band_index} INDEX is "
                    "outside the recorded-frequency table"
                )
            recorded_bands.append(
                RecordedBand(
                    identifier=band_index,
                    frequency_id=recorded_frequency_ids[frequency_slot],
                    polarization=_one(section, f"REC BAND {band_index} POL"),
                    thread_id=thread_ids[band_index],
                    clock_offset_s=float(
                        _one(section, f"CLK OFFSET {frequency_slot} (us)")
                    )
                    * 1e-6,
                    lo_offset_hz=float(
                        _one(section, f"FREQ OFFSET {frequency_slot} (Hz)")
                    ),
                )
            )
        recorded_tuple = tuple(recorded_bands)

        zoom_frequency_count = int(_one(section, "NUM ZOOM FREQS"))
        zoom_frequency_ids = tuple(
            int(_one(section, f"ZOOM FREQ INDEX {slot}"))
            for slot in range(zoom_frequency_count)
        )
        for slot, frequency_id in enumerate(zoom_frequency_ids):
            if not 0 <= frequency_id < frequency_count:
                raise JobLoadError(
                    f"{section.source}: {entity} ZOOM FREQ INDEX {slot} "
                    "is outside the frequency table"
                )
        zoom_band_count = sum(
            int(_one(section, f"NUM ZOOM POLS {slot}"))
            for slot in range(zoom_frequency_count)
        )
        zoom_bands = []
        for zoom_index in range(zoom_band_count):
            frequency_slot = int(
                _one(section, f"ZOOM BAND {zoom_index} INDEX")
            )
            if not 0 <= frequency_slot < zoom_frequency_count:
                raise JobLoadError(
                    f"{section.source}: ZOOM BAND {zoom_index} INDEX is "
                    "outside the zoom-frequency table"
                )
            output_frequency_id = zoom_frequency_ids[frequency_slot]
            polarization = _one(section, f"ZOOM BAND {zoom_index} POL")
            parent_band_id, start_channel = _zoom_parent_and_start(
                frequency_tuple[output_frequency_id],
                recorded_tuple,
                frequency_tuple,
                polarization,
                f"{entity}.zoom_band[{zoom_index}]",
            )
            output_frequency = frequency_tuple[output_frequency_id]
            zoom_bands.append(
                ZoomBand(
                    identifier=recorded_band_count + zoom_index,
                    parent_band_id=parent_band_id,
                    output_frequency_id=output_frequency_id,
                    start_channel=start_channel,
                    channel_count=output_frequency.channel_count,
                    polarization=polarization,
                )
            )

        sampling = _sampling(_one(section, "DATA SAMPLING"), entity)
        source_count = int(_one(input_doc, f"D/STREAM {index} FILES"))
        if source_count < 0:
            raise JobLoadError(
                f"{input_doc.source}: D/STREAM {index} FILES "
                "must be nonnegative"
            )
        source_files = tuple(
            str(
                (
                    input_path.parent
                    / _one(input_doc, f"FILE {index}/{file_index}")
                ).resolve()
            )
            for file_index in range(source_count)
        )
        rate_factor = 2.0 if sampling is SamplingKind.REAL else 1.0
        datastreams.append(
            Datastream(
                identifier=index,
                station_id=int(_one(section, "TELESCOPE INDEX")),
                sampling=sampling,
                quantization_bits=int(_one(section, "QUANTISATION BITS")),
                frame_size_bytes=frame_size_bytes,
                source_files=source_files,
                recorded_bands=recorded_tuple,
                zoom_bands=tuple(zoom_bands),
                sample_rate_hz=frequency_tuple[
                    recorded_frequency_ids[0]
                ].bandwidth_hz
                * rate_factor,
                data_format=data_format,
            )
        )

    baselines = []
    baseline_count = _integer(input_doc, "BASELINE ENTRIES")
    for index, section in enumerate(
        _baseline_sections(input_doc, baseline_count)
    ):
        products: list[BaselineProduct] = []
        product_occurrences: dict[int, int] = {}
        datastream_a_id = int(_one(section, f"D/STREAM A INDEX {index}"))
        datastream_b_id = int(_one(section, f"D/STREAM B INDEX {index}"))
        for side, datastream_id in (
            ("A", datastream_a_id),
            ("B", datastream_b_id),
        ):
            if not 0 <= datastream_id < len(datastreams):
                raise JobLoadError(
                    f"baseline[{index}]: D/STREAM {side} INDEX "
                    f"{datastream_id} is outside the datastream table"
                )
        for frequency_group in range(
            int(_one(section, f"NUM FREQS {index}"))
        ):
            target_key = f"TARGET FREQ {index}/{frequency_group}"
            explicit_targets = section.get_all(target_key)
            if len(explicit_targets) > 1:
                raise JobLoadError(
                    f"{input_doc.source}: field {target_key!r} is repeated"
                )
            explicit_target = (
                int(explicit_targets[0]) if explicit_targets else None
            )
            product_count = int(
                _one(section, f"POL PRODUCTS {index}/{frequency_group}")
            )
            for product_index in range(product_count):
                occurrence = product_occurrences.get(product_index, 0)
                band_a_id = int(
                    _nth(
                        section,
                        f"D/STREAM A BAND {product_index}",
                        occurrence,
                    )
                )
                band_b_id = int(
                    _nth(
                        section,
                        f"D/STREAM B BAND {product_index}",
                        occurrence,
                    )
                )
                if explicit_target is not None:
                    output_frequency_id = explicit_target
                else:
                    # Normalize to A's declared output. validate_job checks
                    # B and the target by spectral grid, not table identity.
                    output_frequency_id = _band_output_frequency(
                        datastreams[datastream_a_id],
                        band_a_id,
                        f"baseline[{index}] product[{len(products)}] A",
                    )
                products.append(
                    BaselineProduct(
                        band_a_id=band_a_id,
                        band_b_id=band_b_id,
                        output_frequency_id=output_frequency_id,
                    )
                )
                product_occurrences[product_index] = occurrence + 1
        baselines.append(
            Baseline(
                identifier=index,
                datastream_a_id=datastream_a_id,
                datastream_b_id=datastream_b_id,
                products=tuple(products),
            )
        )

    scans = []
    # CALC scan offsets are measured from the integral second containing the
    # text job start.  Normalize them to the exact fractional Job.start_time.
    scan_time_origin = start_seconds % Decimal(1)
    for scan_index, section in enumerate(
        _im_scan_sections(im_doc, scan_count)
    ):
        polynomial_count = _integer(section, f"SCAN {scan_index} NUM POLY")
        if polynomial_count <= 0:
            raise JobLoadError(
                f"{im_doc.source}: SCAN {scan_index} NUM POLY must be "
                "positive"
            )
        station_models = []
        for polynomial_index in range(polynomial_count):
            prefix = f"SCAN {scan_index} POLY {polynomial_index}"
            poly_mjd = int(_one(section, f"{prefix} MJD"))
            poly_seconds = Decimal(_one(section, f"{prefix} SEC"))
            epoch_offset = (
                Decimal((poly_mjd - start_mjd) * 86400)
                + poly_seconds
                - start_seconds
            )
            for station_index in range(station_count):
                station_models.append(
                    DelayCoordinateModel(
                        station_id=station_index,
                        epoch_offset_s=epoch_offset,
                        delay_coefficients_s=_coefficients(
                            section,
                            f"SRC 0 ANT {station_index} DELAY (us)",
                            scale=1e-6,
                            occurrence=polynomial_index,
                        ),
                        u_coefficients_m=_coefficients(
                            section,
                            f"SRC 0 ANT {station_index} U (m)",
                            occurrence=polynomial_index,
                        ),
                        v_coefficients_m=_coefficients(
                            section,
                            f"SRC 0 ANT {station_index} V (m)",
                            occurrence=polynomial_index,
                        ),
                        w_coefficients_m=_coefficients(
                            section,
                            f"SRC 0 ANT {station_index} W (m)",
                            occurrence=polynomial_index,
                        ),
                    )
                )
        scans.append(
            Scan(
                identifier=_one(calc_doc, f"SCAN {scan_index} IDENTIFIER"),
                start_offset_s=(
                    Decimal(_one(calc_doc, f"SCAN {scan_index} START (S)"))
                    - scan_time_origin
                ),
                duration_s=Decimal(
                    _one(calc_doc, f"SCAN {scan_index} DUR (S)")
                ),
                source_id=_one(calc_doc, f"SCAN {scan_index} POINTING SRC"),
                configuration_id="0",
                station_models=tuple(station_models),
            )
        )
    # A generated text job may retain a configured feed whose file list is
    # explicitly empty. There is no voltage source for its products. Preserve
    # station and surviving stream identifiers, rather than assigning another
    # feed's recording or making an empty source pass normalized validation.
    # Nonempty file lists remain strict: missing/unreadable paths are not
    # treated as unavailable feeds here.
    unavailable = {
        stream.identifier for stream in datastreams if not stream.source_files
    }
    if unavailable:
        warnings.warn(
            "Text job declares no recordings for datastreams "
            + ", ".join(map(str, sorted(unavailable)))
            + "; omitting their unavailable products",
            RuntimeWarning,
            stacklevel=2,
        )
        datastreams = [
            stream
            for stream in datastreams
            if stream.identifier not in unavailable
        ]
        baselines = [
            baseline
            for baseline in baselines
            if baseline.datastream_a_id not in unavailable
            and baseline.datastream_b_id not in unavailable
        ]

    job = Job(
        buffered_reader=_buffered_reader_geometry(input_doc),
        start_time=MJDTime(start_mjd, start_seconds),
        execution_duration_s=Decimal(_one(input_doc, "EXECUTE TIME (SEC)")),
        integration_duration_s=Decimal(_one(input_doc, "INT TIME (SEC)")),
        subintegration_duration_s=Decimal(
            _one(input_doc, "SUBINT NANOSECONDS")
        )
        / Decimal(1_000_000_000),
        fringe_approximation_order=int(_one(input_doc, "FRINGE ROTN ORDER")),
        stations=tuple(stations),
        frequencies=frequency_tuple,
        datastreams=tuple(datastreams),
        baselines=tuple(baselines),
        scans=tuple(scans),
        write_autocorrelations=_boolean(_one(input_doc, "WRITE AUTOCORRS")),
        output=OutputConfiguration(OutputFormat.SWIN),
        unsupported_features=UnsupportedFeatures(
            pulsar_binning=_boolean(_one(input_doc, "PULSAR BINNING")),
            phased_array_output=_boolean(_one(input_doc, "PHASED ARRAY")),
        ),
    )
    validate_job(job)
    return job


def declared_output_path(input_path: str | Path) -> Path:
    document = parse_input_file(input_path)
    return (
        Path(input_path).resolve().parent / _one(document, "OUTPUT FILENAME")
    ).resolve()


__all__ = ["declared_output_path", "load_interop_job"]
