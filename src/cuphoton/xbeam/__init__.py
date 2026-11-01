# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Public correlation APIs, loaded on demand."""

from importlib import import_module

from cuphoton import __version__

_EXPORTS = {
    "BackendUnavailableError": ("errors", "BackendUnavailableError"),
    "XBeamError": ("errors", "XBeamError"),
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
