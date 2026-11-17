# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import cmath
import itertools
import math
from dataclasses import replace
from decimal import Decimal

import numpy as np
import pytest

from cuphoton.xbeam.adapters.radio.closure import compute_closures
from cuphoton.xbeam.adapters.radio.models import VisibilityRecord
from cuphoton.xbeam.adapters.radio.output import iter_xbeam_v1, write_xbeam_v1
from cuphoton.xbeam.adapters.radio.swin import (
    from_visibility,
    iter_swin,
    write_swin,
)
from cuphoton.xbeam.core.products import cross_power

TRIANGLE = ("A", "B", "C")
QUADRANGLE = ("A", "B", "C", "D")


def record(a, b, spectrum, **changes):
    return replace(
        VisibilityRecord(
            antenna_a=a,
            antenna_b=b,
            mjd=60000,
            seconds_of_day=Decimal("123.000000000001"),
            source_id="source",
            configuration_id="config",
            output_frequency_id=0,
            polarization_a="R",
            polarization_b="R",
            weight=0.75,
            u_m=0.0,
            v_m=0.0,
            w_m=0.0,
            spectrum=np.asarray(spectrum, dtype=np.complex128),
        ),
        **changes,
    )


def closures(records):
    return compute_closures(
        records, triangles=[TRIANGLE], quadrangles=[QUADRANGLE]
    )


def station_voltages():
    rng = np.random.default_rng(71)
    sources = rng.normal(size=(2, 16, 8)) + 1j * rng.normal(size=(2, 16, 8))
    response = np.array([[1, 0.2j], [0.7j, 1], [1 - 0.3j, -0.8j], [-1, 0.6]])
    return np.einsum("as,sbn->abn", response, sources)


def correlate(stations):
    spectra = np.fft.fft(stations, axis=-1)
    return [
        record(
            chr(65 + a),
            chr(65 + b),
            cross_power(spectra[a], spectra[b], np).mean(axis=0),
        )
        for a, b in itertools.combinations(range(4), 2)
    ]


def direct_dft_oracle(stations):
    """Scalar DFT and correlation, independent of FFT/cross_power and API."""
    phase, amplitude = [], []
    for channel in range(stations.shape[-1]):
        transforms = [
            [
                sum(
                    sample
                    * cmath.exp(-2j * math.pi * channel * n / len(block))
                    for n, sample in enumerate(block)
                )
                for block in station
            ]
            for station in stations
        ]

        def visibility(a, b, transforms=transforms):
            return sum(
                x * y.conjugate()
                for x, y in zip(transforms[a], transforms[b], strict=True)
            ) / len(transforms[a])

        phase.append(
            cmath.phase(
                visibility(0, 1) * visibility(1, 2) * visibility(2, 0)
            )
        )
        amplitude.append(
            math.log(
                abs(
                    visibility(0, 1)
                    * visibility(2, 3)
                    / (visibility(0, 2) * visibility(1, 3))
                )
            )
        )
    return np.array(phase), np.array(amplitude)


def test_nontrivial_closures_match_direct_dft_and_cancel_station_gains():
    stations = station_voltages()
    expected = direct_dft_oracle(stations)
    # These fixtures would expose an all-conjugated correlator result.
    assert np.max(np.abs(expected[0])) > 0.2
    assert np.max(np.abs(expected[1])) > 0.2
    gain = np.array([2 * np.exp(0.2j), 0.3j, -1.5, np.exp(-1.2j)])
    original = closures(correlate(stations))
    corrupted = closures(correlate(stations * gain[:, None, None]))
    for baseline, gained, oracle in zip(
        original, corrupted, expected, strict=True
    ):
        assert baseline.valid.all() and gained.valid.all()
        np.testing.assert_allclose(
            baseline.values, oracle, rtol=0, atol=2e-14
        )
        np.testing.assert_allclose(gained.values, oracle, rtol=0, atol=2e-14)


@pytest.mark.parametrize("format_name", ["xbeam-v1", "swin"])
def test_saved_outputs_and_reversed_edges_preserve_closures(
    tmp_path, format_name
):
    records = correlate(station_voltages())
    expected = closures(records)
    reversed_edges = [
        replace(
            r,
            antenna_a=r.antenna_b,
            antenna_b=r.antenna_a,
            spectrum=r.spectrum.conj(),
        )
        if i % 2
        else r
        for i, r in enumerate(records)
    ][::-1]
    if format_name == "xbeam-v1":
        write_xbeam_v1(tmp_path, reversed_edges)
        actual = closures(iter_xbeam_v1(tmp_path))
        assert actual[0].key.seconds_of_day == records[0].seconds_of_day
    else:
        mapping = {name: i for i, name in enumerate(QUADRANGLE)}
        path = tmp_path / "visibilities.swin"
        write_swin(
            path,
            [
                from_visibility(
                    r,
                    mapping,
                    source_indices={"source": 0},
                    configuration_indices={"config": 0},
                )
                for r in reversed_edges
            ],
        )
        actual = compute_closures(
            iter_swin(path, 8),
            triangles=[(0, 1, 2)],
            quadrangles=[(0, 1, 2, 3)],
        )
    for result, reference in zip(actual, expected, strict=True):
        assert result.valid.all()
        np.testing.assert_allclose(
            result.values, reference.values, rtol=0, atol=2e-7
        )


def test_triangle_order_changes_sign_and_quad_order_changes_ratio():
    records = correlate(station_voltages())
    expected = closures(records)
    actual = compute_closures(
        records,
        triangles=[("A", "C", "B")],
        quadrangles=[("A", "C", "B", "D")],
    )
    np.testing.assert_allclose(
        actual[0].values, -expected[0].values, rtol=0, atol=1e-14
    )
    np.testing.assert_allclose(
        actual[1].values, -expected[1].values, rtol=0, atol=1e-14
    )


@pytest.mark.parametrize("bad", [0, np.nan, complex(np.inf, 0)])
def test_undefined_channels_are_masked_without_discarding_good_channels(bad):
    records = correlate(station_voltages())
    records[0].spectrum[2] = bad
    with np.errstate(all="raise"):
        actual = closures(records)
    for result in actual:
        assert not result.valid[2]
        assert np.isnan(result.values[2])
        assert result.valid[[0, 1, 3, 4, 5, 6, 7]].all()


@pytest.mark.parametrize("weight", [0, -1, np.nan, np.inf])
def test_invalid_exposure_invalidates_closures(weight):
    records = correlate(station_voltages())
    records[0] = replace(records[0], weight=weight)
    for result in closures(records):
        assert not result.valid.any()
        assert np.isnan(result.values).all()


def test_missing_edge_marks_only_dependent_closures_invalid():
    # C-D occurs only in the requested quadrangle, not A-B-C.
    phase, amplitude = closures(correlate(station_voltages())[:-1])
    assert phase.valid.all()
    assert not amplitude.valid.any()
    assert np.isnan(amplitude.values).all()


@pytest.mark.parametrize(
    "field,value",
    [
        ("mjd", 60001),
        ("seconds_of_day", Decimal("123.000000000002")),
        ("source_id", "different"),
        ("configuration_id", "different"),
        ("output_frequency_id", 1),
    ],
)
def test_distinct_metadata_never_fills_missing_edges(field, value):
    records = correlate(station_voltages())
    records[0] = replace(records[0], **{field: value})
    actual = closures(records)
    assert len({r.key for r in actual}) == 2
    assert not any(r.valid.any() for r in actual)


def test_hands_are_separate_and_crosshands_and_autocorrelations_are_ignored():
    records = correlate(station_voltages())
    ignored = [
        replace(records[0], polarization_b="L"),
        replace(records[0], antenna_b="A"),
    ]
    partial_l = replace(records[0], polarization_a="L", polarization_b="L")
    actual = closures([*records, *ignored, partial_l])
    assert len(actual) == 4
    assert [r.key.polarization for r in actual] == ["R", "R", "L", "L"]
    assert actual[0].valid.all() and actual[1].valid.all()
    assert not actual[2].valid.any() and not actual[3].valid.any()


def test_swin_pulsar_bins_remain_separate():
    mapping = {name: i for i, name in enumerate(QUADRANGLE)}
    records = [
        from_visibility(
            r,
            mapping,
            source_indices={"source": 0},
            configuration_indices={"config": 0},
        )
        for r in correlate(station_voltages())
    ]
    records[0] = replace(records[0], pulsar_bin=1)
    actual = compute_closures(records, triangles=[(0, 1, 2)])
    assert {r.key.pulsar_bin for r in actual} == {0, 1}
    assert not any(r.valid.any() for r in actual)


@pytest.mark.parametrize("reverse", [False, True])
def test_duplicate_edges_are_rejected(reverse):
    records = correlate(station_voltages())
    duplicate = records[0]
    if reverse:
        duplicate = replace(
            duplicate,
            antenna_a="B",
            antenna_b="A",
            spectrum=duplicate.spectrum.conj(),
        )
    with pytest.raises(ValueError, match="duplicate baseline"):
        closures([*records, duplicate])


def test_channel_grids_and_distinct_stations_are_required():
    records = correlate(station_voltages())
    records[0] = replace(records[0], spectrum=records[0].spectrum[:-1])
    with pytest.raises(ValueError, match="inconsistent channel counts"):
        closures(records)
    with pytest.raises(ValueError, match="3 distinct"):
        compute_closures([], triangles=[("A", "B", "A")])


@pytest.mark.parametrize("scale", [1e-250, 1e250])
def test_extreme_finite_amplitudes_do_not_overflow_products(scale):
    records = correlate(station_voltages())
    expected = closures(records)
    for r in records:
        r.spectrum[:] *= scale
    with np.errstate(all="raise"):
        actual = closures(records)
    for result, reference in zip(actual, expected, strict=True):
        assert result.valid.all()
        np.testing.assert_allclose(
            result.values, reference.values, rtol=0, atol=3e-13
        )
