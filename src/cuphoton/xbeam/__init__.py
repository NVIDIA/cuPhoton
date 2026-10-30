# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Public correlation APIs, loaded on demand."""

from importlib import import_module

from cuphoton import __version__

_EXPORTS = {
    "BackendUnavailableError": ("errors", "BackendUnavailableError"),
    "ConfigurationError": ("errors", "ConfigurationError"),
    "JobLoadError": ("errors", "JobLoadError"),
    "UnsupportedFeatureError": ("errors", "UnsupportedFeatureError"),
    "UnsupportedProfileError": ("errors", "UnsupportedProfileError"),
    "XBeamError": ("errors", "XBeamError"),
    "SCHEMA_VERSION": ("adapters.radio.models", "SCHEMA_VERSION"),
    "Baseline": ("adapters.radio.models", "Baseline"),
    "BaselineProduct": ("adapters.radio.models", "BaselineProduct"),
    "BufferedReaderGeometry": (
        "adapters.radio.models",
        "BufferedReaderGeometry",
    ),
    "ClockPolynomial": ("adapters.radio.models", "ClockPolynomial"),
    "DataSourceKind": ("adapters.radio.models", "DataSourceKind"),
    "Datastream": ("adapters.radio.models", "Datastream"),
    "DelayCoordinateModel": ("adapters.radio.models", "DelayCoordinateModel"),
    "Frequency": ("adapters.radio.models", "Frequency"),
    "Job": ("adapters.radio.models", "Job"),
    "MJDTime": ("adapters.radio.models", "MJDTime"),
    "OutputConfiguration": ("adapters.radio.models", "OutputConfiguration"),
    "OutputFormat": ("adapters.radio.models", "OutputFormat"),
    "RecordedBand": ("adapters.radio.models", "RecordedBand"),
    "RecordingSource": ("adapters.radio.models", "RecordingSource"),
    "SamplingKind": ("adapters.radio.models", "SamplingKind"),
    "Scan": ("adapters.radio.models", "Scan"),
    "Sideband": ("adapters.radio.models", "Sideband"),
    "Station": ("adapters.radio.models", "Station"),
    "StationSpectralBlock": ("adapters.radio.models", "StationSpectralBlock"),
    "UnsupportedFeatures": ("adapters.radio.models", "UnsupportedFeatures"),
    "VisibilityRecord": ("adapters.radio.models", "VisibilityRecord"),
    "VisibilitySink": ("adapters.radio.models", "VisibilitySink"),
    "VoltageBatch": ("adapters.radio.models", "VoltageBatch"),
    "ZoomBand": ("adapters.radio.models", "ZoomBand"),
    "dump_job": ("adapters.radio.models", "dump_job"),
    "job_from_dict": ("adapters.radio.models", "job_from_dict"),
    "job_to_dict": ("adapters.radio.models", "job_to_dict"),
    "load_job": ("adapters.radio.models", "load_job"),
    "load_job_schema": ("adapters.radio.models", "load_job_schema"),
    "validate_job": ("validation", "validate_job"),
    "CorrelationSpec": ("core.series", "CorrelationSpec"),
    "SeriesResult": ("core.series", "SeriesResult"),
    "SeriesShard": ("core.series", "SeriesShard"),
    "correlate_series": ("core.series", "correlate_series"),
    "plan_series_shards": ("core.series", "plan_series_shards"),
    "ArraySource": ("adapters.arrays", "ArraySource"),
    "Hdf5Source": ("adapters.hdf5", "Hdf5Source"),
}

__all__ = ["__version__", *_EXPORTS]


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(f".{module}", __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
