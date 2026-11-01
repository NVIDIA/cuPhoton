# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Typed data contracts for normalized correlation jobs."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum, StrEnum
from importlib.resources import files
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from cuphoton.xbeam.errors import JobLoadError

SCHEMA_VERSION = 1


class Sideband(StrEnum):
    USB = "USB"
    LSB = "LSB"


class SamplingKind(StrEnum):
    REAL = "real"
    COMPLEX = "complex"
    COMPLEX_DSB = "complex_dsb"


class DataSourceKind(StrEnum):
    FILE = "file"


class OutputFormat(StrEnum):
    NATIVE = "xbeam-v1"
    SWIN = "swin"


class AutocorrelationMode(StrEnum):
    """Channel reduction used for parallel-hand autocorrelations."""

    AVERAGE = "average"
    TRUNCATION = "truncation"


@dataclass(frozen=True)
class MJDTime:
    """Lossless boundary timestamp represented as MJD and decimal seconds."""

    mjd: int
    seconds_of_day: Decimal


@dataclass(frozen=True)
class ClockPolynomial:
    epoch: MJDTime
    coefficients_s: tuple[float, ...]


@dataclass(frozen=True)
class Station:
    identifier: int
    name: str
    clock: ClockPolynomial


@dataclass(frozen=True)
class Frequency:
    identifier: int
    band_edge_hz: float
    bandwidth_hz: float
    sideband: Sideband
    channel_count: int
    channels_to_average: int = 1
    oversampling: int = 1
    decimation: int = 1


@dataclass(frozen=True)
class RecordedBand:
    identifier: int
    frequency_id: int
    polarization: str
    thread_id: int = 0
    clock_offset_s: float = 0.0
    lo_offset_hz: float = 0.0


@dataclass(frozen=True)
class ZoomBand:
    identifier: int
    parent_band_id: int
    output_frequency_id: int
    start_channel: int
    channel_count: int
    polarization: str


@dataclass(frozen=True)
class Datastream:
    identifier: int
    station_id: int
    sampling: SamplingKind
    quantization_bits: int
    frame_size_bytes: int
    source_files: tuple[str, ...]
    recorded_bands: tuple[RecordedBand, ...]
    zoom_bands: tuple[ZoomBand, ...] = ()
    sample_rate_hz: float | None = None
    data_format: str = "VDIF"
    data_source: DataSourceKind = DataSourceKind.FILE


@dataclass(frozen=True)
class BaselineProduct:
    band_a_id: int
    band_b_id: int
    output_frequency_id: int


@dataclass(frozen=True)
class Baseline:
    identifier: int
    datastream_a_id: int
    datastream_b_id: int
    products: tuple[BaselineProduct, ...]


@dataclass(frozen=True)
class DelayCoordinateModel:
    station_id: int
    epoch_offset_s: Decimal
    delay_coefficients_s: tuple[float, ...]
    u_coefficients_m: tuple[float, ...]
    v_coefficients_m: tuple[float, ...]
    w_coefficients_m: tuple[float, ...]


@dataclass(frozen=True)
class Scan:
    identifier: str
    start_offset_s: Decimal
    duration_s: Decimal
    source_id: str
    configuration_id: str
    station_models: tuple[DelayCoordinateModel, ...]


@dataclass(frozen=True)
class UnsupportedFeatures:
    pulsar_binning: bool = False
    multiple_phase_centres: bool = False
    phased_array_output: bool = False
    phase_calibration: bool = False
    system_temperature_calibration: bool = False
    switched_power: bool = False
    polarization_conversion: bool = False
    output_band_stitching: bool = False
    spectral_kurtosis: bool = False


@dataclass(frozen=True)
class OutputConfiguration:
    format: OutputFormat = OutputFormat.NATIVE
    spectral_placement: int = 0


@dataclass(frozen=True)
class BufferedReaderGeometry:
    """Reader settings needed to reproduce buffered window ownership.

    Absence of these settings means that no reader behavior was specified;
    it must not select an implicit calibrated geometry.
    """

    buffer_factor: int
    data_segments: int
    buffered_ffts: int


@dataclass(frozen=True)
class Job:
    start_time: MJDTime
    execution_duration_s: Decimal
    integration_duration_s: Decimal
    subintegration_duration_s: Decimal
    fringe_approximation_order: int
    stations: tuple[Station, ...]
    frequencies: tuple[Frequency, ...]
    datastreams: tuple[Datastream, ...]
    baselines: tuple[Baseline, ...]
    scans: tuple[Scan, ...]
    write_autocorrelations: bool = False
    output: OutputConfiguration = field(default_factory=OutputConfiguration)
    unsupported_features: UnsupportedFeatures = field(
        default_factory=UnsupportedFeatures
    )
    schema_version: int = SCHEMA_VERSION
    buffered_reader: BufferedReaderGeometry | None = None


@dataclass(frozen=True)
class PackedVoltageLayout:
    """Layout metadata for frame-packed two-bit voltage samples.

    When present on :class:`VoltageBatch`, ``samples`` is a contiguous uint8
    array containing payload bytes only and ``validity`` has one entry per
    frame.  Without it, samples and validity are both sample-granular.
    """

    frame_payload_bytes: int
    samples_per_frame: int
    channel_count: int
    channel_index: int
    complex_data: bool


@dataclass(frozen=True)
class VoltageBatch:
    samples: Any
    start_time: MJDTime
    sample_rate_hz: float
    band_id: int
    polarization: str
    validity: Any
    packed_layout: PackedVoltageLayout | None = None


@runtime_checkable
class RecordingSource(Protocol):
    """Yield normalized voltage batches without exposing storage layout."""

    def batches(self, datastream: Datastream) -> Iterable[VoltageBatch]: ...


@dataclass(frozen=True)
class StationSpectralBlock:
    spectrum: Any
    channel_zero_hz: float
    channel_spacing_hz: float
    validity_weight: float
    station_id: int
    polarization: str
    frequency_id: int


@dataclass(frozen=True)
class VisibilityRecord:
    antenna_a: str
    antenna_b: str
    mjd: int
    seconds_of_day: Decimal
    source_id: str
    configuration_id: str
    output_frequency_id: int
    polarization_a: str
    polarization_b: str
    weight: float
    u_m: float
    v_m: float
    w_m: float
    spectrum: Any


@runtime_checkable
class VisibilitySink(Protocol):
    def write(self, record: VisibilityRecord) -> None: ...

    def close(self) -> None: ...


def _decimal(value: Any, path: str) -> Decimal:
    if isinstance(value, bool):
        raise JobLoadError(f"{path}: expected a decimal number")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise JobLoadError(f"{path}: expected a decimal number") from exc


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise JobLoadError(f"{path}: expected an object")
    return value


def _sequence(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise JobLoadError(f"{path}: expected an array")
    return value


def _strict(data: Mapping[str, Any], allowed: set[str], path: str) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise JobLoadError(f"{path}: unknown field {unknown[0]!r}")


def _enum(enum_type: type[Enum], value: Any, path: str) -> Any:
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        choices = ", ".join(repr(item.value) for item in enum_type)
        raise JobLoadError(f"{path}: expected one of {choices}") from exc


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise JobLoadError(f"{path}: expected a string")
    return value


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise JobLoadError(f"{path}: expected a boolean")
    return value


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool):
        raise JobLoadError(f"{path}: expected an integer")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise JobLoadError(f"{path}: expected an integer") from exc
    # Preserve integer strings and integral numbers without truncating them.
    if not isinstance(value, str) and value != result:
        raise JobLoadError(f"{path}: expected an integer")
    return result


def _time(value: Any, path: str) -> MJDTime:
    data = _mapping(value, path)
    _strict(data, {"mjd", "seconds_of_day"}, path)
    try:
        return MJDTime(
            _integer(data["mjd"], f"{path}.mjd"),
            _decimal(data["seconds_of_day"], f"{path}.seconds_of_day"),
        )
    except KeyError as exc:
        raise JobLoadError(f"{path}: missing field {exc.args[0]!r}") from exc


def _float_tuple(value: Any, path: str) -> tuple[float, ...]:
    return tuple(float(item) for item in _sequence(value, path))


def job_from_dict(document: Mapping[str, Any]) -> Job:
    """Decode the strict normalized schema without static validation."""

    root = _mapping(document, "job")
    allowed = {
        "schema_version",
        "start_time",
        "execution_duration_s",
        "integration_duration_s",
        "subintegration_duration_s",
        "fringe_approximation_order",
        "stations",
        "frequencies",
        "datastreams",
        "baselines",
        "scans",
        "write_autocorrelations",
        "output",
        "unsupported_features",
        "buffered_reader",
    }
    _strict(root, allowed, "job")
    try:
        stations = []
        for index, raw in enumerate(
            _sequence(root["stations"], "job.stations")
        ):
            path = f"job.stations[{index}]"
            data = _mapping(raw, path)
            _strict(data, {"identifier", "name", "clock"}, path)
            clock_data = _mapping(data["clock"], f"{path}.clock")
            _strict(clock_data, {"epoch", "coefficients_s"}, f"{path}.clock")
            stations.append(
                Station(
                    _integer(data["identifier"], f"{path}.identifier"),
                    _string(data["name"], f"{path}.name"),
                    ClockPolynomial(
                        _time(clock_data["epoch"], f"{path}.clock.epoch"),
                        _float_tuple(
                            clock_data["coefficients_s"],
                            f"{path}.clock.coefficients_s",
                        ),
                    ),
                )
            )

        frequencies = []
        for index, raw in enumerate(
            _sequence(root["frequencies"], "job.frequencies")
        ):
            path = f"job.frequencies[{index}]"
            data = _mapping(raw, path)
            _strict(
                data,
                {
                    "identifier",
                    "band_edge_hz",
                    "bandwidth_hz",
                    "sideband",
                    "channel_count",
                    "channels_to_average",
                    "oversampling",
                    "decimation",
                },
                path,
            )
            frequencies.append(
                Frequency(
                    _integer(data["identifier"], f"{path}.identifier"),
                    float(data["band_edge_hz"]),
                    float(data["bandwidth_hz"]),
                    _enum(Sideband, data["sideband"], f"{path}.sideband"),
                    _integer(data["channel_count"], f"{path}.channel_count"),
                    _integer(
                        data.get("channels_to_average", 1),
                        f"{path}.channels_to_average",
                    ),
                    _integer(
                        data.get("oversampling", 1), f"{path}.oversampling"
                    ),
                    _integer(data.get("decimation", 1), f"{path}.decimation"),
                )
            )

        datastreams = []
        for index, raw in enumerate(
            _sequence(root["datastreams"], "job.datastreams")
        ):
            path = f"job.datastreams[{index}]"
            data = _mapping(raw, path)
            _strict(
                data,
                {
                    "identifier",
                    "station_id",
                    "sampling",
                    "quantization_bits",
                    "frame_size_bytes",
                    "source_files",
                    "recorded_bands",
                    "zoom_bands",
                    "sample_rate_hz",
                    "data_format",
                    "data_source",
                },
                path,
            )
            recorded = []
            for band_index, raw_band in enumerate(
                _sequence(data["recorded_bands"], f"{path}.recorded_bands")
            ):
                band_path = f"{path}.recorded_bands[{band_index}]"
                band = _mapping(raw_band, band_path)
                _strict(
                    band,
                    {
                        "identifier",
                        "frequency_id",
                        "polarization",
                        "thread_id",
                        "clock_offset_s",
                        "lo_offset_hz",
                    },
                    band_path,
                )
                recorded.append(
                    RecordedBand(
                        _integer(
                            band["identifier"], f"{band_path}.identifier"
                        ),
                        _integer(
                            band["frequency_id"], f"{band_path}.frequency_id"
                        ),
                        _string(
                            band["polarization"], f"{band_path}.polarization"
                        ),
                        _integer(
                            band.get("thread_id", 0), f"{band_path}.thread_id"
                        ),
                        float(band.get("clock_offset_s", 0.0)),
                        float(band.get("lo_offset_hz", 0.0)),
                    )
                )
            zooms = []
            for zoom_index, raw_zoom in enumerate(
                _sequence(data.get("zoom_bands", []), f"{path}.zoom_bands")
            ):
                zoom_path = f"{path}.zoom_bands[{zoom_index}]"
                zoom = _mapping(raw_zoom, zoom_path)
                _strict(
                    zoom,
                    {
                        "identifier",
                        "parent_band_id",
                        "output_frequency_id",
                        "start_channel",
                        "channel_count",
                        "polarization",
                    },
                    zoom_path,
                )
                zooms.append(
                    ZoomBand(
                        _integer(
                            zoom["identifier"], f"{zoom_path}.identifier"
                        ),
                        _integer(
                            zoom["parent_band_id"],
                            f"{zoom_path}.parent_band_id",
                        ),
                        _integer(
                            zoom["output_frequency_id"],
                            f"{zoom_path}.output_frequency_id",
                        ),
                        _integer(
                            zoom["start_channel"],
                            f"{zoom_path}.start_channel",
                        ),
                        _integer(
                            zoom["channel_count"],
                            f"{zoom_path}.channel_count",
                        ),
                        _string(
                            zoom["polarization"], f"{zoom_path}.polarization"
                        ),
                    )
                )
            datastreams.append(
                Datastream(
                    _integer(data["identifier"], f"{path}.identifier"),
                    _integer(data["station_id"], f"{path}.station_id"),
                    _enum(SamplingKind, data["sampling"], f"{path}.sampling"),
                    _integer(
                        data["quantization_bits"], f"{path}.quantization_bits"
                    ),
                    _integer(
                        data["frame_size_bytes"], f"{path}.frame_size_bytes"
                    ),
                    tuple(
                        _string(item, f"{path}.source_files")
                        for item in _sequence(
                            data["source_files"], f"{path}.source_files"
                        )
                    ),
                    tuple(recorded),
                    tuple(zooms),
                    None
                    if data.get("sample_rate_hz") is None
                    else float(data["sample_rate_hz"]),
                    _string(
                        data.get("data_format", "VDIF"), f"{path}.data_format"
                    ),
                    _enum(
                        DataSourceKind,
                        data.get("data_source", "file"),
                        f"{path}.data_source",
                    ),
                )
            )

        baselines = []
        for index, raw in enumerate(
            _sequence(root["baselines"], "job.baselines")
        ):
            path = f"job.baselines[{index}]"
            data = _mapping(raw, path)
            _strict(
                data,
                {
                    "identifier",
                    "datastream_a_id",
                    "datastream_b_id",
                    "products",
                },
                path,
            )
            products = []
            for product_index, raw_product in enumerate(
                _sequence(data["products"], f"{path}.products")
            ):
                product_path = f"{path}.products[{product_index}]"
                product = _mapping(raw_product, product_path)
                _strict(
                    product,
                    {"band_a_id", "band_b_id", "output_frequency_id"},
                    product_path,
                )
                products.append(
                    BaselineProduct(
                        _integer(
                            product["band_a_id"], f"{product_path}.band_a_id"
                        ),
                        _integer(
                            product["band_b_id"], f"{product_path}.band_b_id"
                        ),
                        _integer(
                            product["output_frequency_id"],
                            f"{product_path}.output_frequency_id",
                        ),
                    )
                )
            baselines.append(
                Baseline(
                    _integer(data["identifier"], f"{path}.identifier"),
                    _integer(
                        data["datastream_a_id"], f"{path}.datastream_a_id"
                    ),
                    _integer(
                        data["datastream_b_id"], f"{path}.datastream_b_id"
                    ),
                    tuple(products),
                )
            )

        scans = []
        for index, raw in enumerate(_sequence(root["scans"], "job.scans")):
            path = f"job.scans[{index}]"
            data = _mapping(raw, path)
            _strict(
                data,
                {
                    "identifier",
                    "start_offset_s",
                    "duration_s",
                    "source_id",
                    "configuration_id",
                    "station_models",
                },
                path,
            )
            models = []
            for model_index, raw_model in enumerate(
                _sequence(data["station_models"], f"{path}.station_models")
            ):
                model_path = f"{path}.station_models[{model_index}]"
                model = _mapping(raw_model, model_path)
                _strict(
                    model,
                    {
                        "station_id",
                        "epoch_offset_s",
                        "delay_coefficients_s",
                        "u_coefficients_m",
                        "v_coefficients_m",
                        "w_coefficients_m",
                    },
                    model_path,
                )
                models.append(
                    DelayCoordinateModel(
                        _integer(
                            model["station_id"], f"{model_path}.station_id"
                        ),
                        _decimal(
                            model["epoch_offset_s"],
                            f"{model_path}.epoch_offset_s",
                        ),
                        _float_tuple(
                            model["delay_coefficients_s"],
                            f"{model_path}.delay_coefficients_s",
                        ),
                        _float_tuple(
                            model["u_coefficients_m"],
                            f"{model_path}.u_coefficients_m",
                        ),
                        _float_tuple(
                            model["v_coefficients_m"],
                            f"{model_path}.v_coefficients_m",
                        ),
                        _float_tuple(
                            model["w_coefficients_m"],
                            f"{model_path}.w_coefficients_m",
                        ),
                    )
                )
            scans.append(
                Scan(
                    _string(data["identifier"], f"{path}.identifier"),
                    _decimal(
                        data["start_offset_s"], f"{path}.start_offset_s"
                    ),
                    _decimal(data["duration_s"], f"{path}.duration_s"),
                    _string(data["source_id"], f"{path}.source_id"),
                    _string(
                        data["configuration_id"], f"{path}.configuration_id"
                    ),
                    tuple(models),
                )
            )

        output_data = _mapping(root.get("output", {}), "job.output")
        _strict(output_data, {"format", "spectral_placement"}, "job.output")
        output = OutputConfiguration(
            _enum(
                OutputFormat,
                output_data.get("format", "xbeam-v1"),
                "job.output.format",
            ),
            _integer(
                output_data.get("spectral_placement", 0),
                "job.output.spectral_placement",
            ),
        )
        feature_data = _mapping(
            root.get("unsupported_features", {}), "job.unsupported_features"
        )
        feature_names = set(UnsupportedFeatures.__dataclass_fields__)
        _strict(feature_data, feature_names, "job.unsupported_features")
        features = UnsupportedFeatures(
            **{
                name: _boolean(
                    feature_data.get(name, False),
                    f"job.unsupported_features.{name}",
                )
                for name in feature_names
            }
        )

        reader = None
        if root.get("buffered_reader") is not None:
            reader_data = _mapping(
                root["buffered_reader"], "job.buffered_reader"
            )
            reader_fields = {
                "buffer_factor",
                "data_segments",
                "buffered_ffts",
            }
            _strict(reader_data, reader_fields, "job.buffered_reader")
            for name in reader_fields:
                value = reader_data[name]
                if type(value) is not int or value <= 0:
                    raise JobLoadError(
                        f"job.buffered_reader.{name}: "
                        "must be a positive integer"
                    )
            reader = BufferedReaderGeometry(**reader_data)

        return Job(
            _time(root["start_time"], "job.start_time"),
            _decimal(
                root["execution_duration_s"], "job.execution_duration_s"
            ),
            _decimal(
                root["integration_duration_s"], "job.integration_duration_s"
            ),
            _decimal(
                root["subintegration_duration_s"],
                "job.subintegration_duration_s",
            ),
            _integer(
                root["fringe_approximation_order"],
                "job.fringe_approximation_order",
            ),
            tuple(stations),
            tuple(frequencies),
            tuple(datastreams),
            tuple(baselines),
            tuple(scans),
            _boolean(
                root.get("write_autocorrelations", False),
                "job.write_autocorrelations",
            ),
            output,
            features,
            _integer(
                root["schema_version"],
                "job.schema_version",
            ),
            reader,
        )
    except KeyError as exc:
        raise JobLoadError(f"job: missing field {exc.args[0]!r}") from exc
    except (TypeError, ValueError) as exc:
        if isinstance(exc, JobLoadError):
            raise
        raise JobLoadError(f"job: invalid value ({exc})") from exc


def _json_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    return value


def job_to_dict(job: Job) -> dict[str, Any]:
    """Encode a job using the normalized versioned JSON representation."""

    result = _json_value(asdict(job))
    if job.buffered_reader is None:
        result.pop("buffered_reader")
    return result


def load_job(path: str | Path) -> Job:
    """Load and statically validate a normalized JSON job."""

    try:
        with Path(path).open("r", encoding="utf-8") as stream:
            document = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise JobLoadError(f"cannot load job {path!s}: {exc}") from exc
    job = job_from_dict(document)
    from cuphoton.xbeam.validation import validate_job

    validate_job(job)
    return job


def dump_job(job: Job, path: str | Path) -> None:
    """Write a normalized JSON job after static validation."""

    from cuphoton.xbeam.validation import validate_job

    validate_job(job)
    try:
        with Path(path).open("w", encoding="utf-8") as stream:
            json.dump(job_to_dict(job), stream, indent=2, sort_keys=True)
            stream.write("\n")
    except (OSError, UnicodeError) as exc:
        raise JobLoadError(f"cannot write job {path!s}: {exc}") from exc


def load_job_schema() -> dict[str, Any]:
    """Return the bundled JSON Schema for normalized job version 1."""

    resource = files("cuphoton.xbeam").joinpath("schemas/job-v1.schema.json")
    return json.loads(resource.read_text(encoding="utf-8"))
