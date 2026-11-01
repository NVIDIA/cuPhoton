# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _load_benchmark_module():
    path = (
        Path(__file__).parents[2]
        / "examples"
        / "xbeam"
        / "benchmark_visibility.py"
    )
    spec = importlib.util.spec_from_file_location(
        "benchmark_visibility", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


benchmark = _load_benchmark_module()


def test_visibility_benchmark_writes_nested_output(
    tmp_path, monkeypatch, capsys
):
    report = {"parity": {"passed": True}}
    monkeypatch.setattr(benchmark, "_run", lambda arguments: report)
    output = tmp_path / "reports" / "visibility" / "result.json"

    assert benchmark.main(["--output", str(output)]) == 0

    assert json.loads(output.read_text(encoding="utf-8")) == report
    assert json.loads(capsys.readouterr().out) == report


def test_visibility_benchmark_product_selection_covers_sparse_and_full_sets():
    assert benchmark._product_bands(1) == ((0, 1),)
    assert benchmark._product_bands(2) == ((0, 2), (1, 3))
    assert benchmark._product_bands(4) == (
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
    )
    assert benchmark._product_bands(4, "distinct") == (
        (0, 1),
        (2, 3),
        (4, 5),
        (6, 7),
    )
    assert benchmark._band_count(benchmark._product_bands(4)) == 4
    assert benchmark._band_count(benchmark._product_bands(4, "distinct")) == 8
    with pytest.raises(ValueError, match="between one and four"):
        benchmark._product_bands(5)
    with pytest.raises(ValueError, match="dual-pol or distinct"):
        benchmark._product_bands(4, "invalid")


def test_visibility_expected_values_preserve_complex_and_weight_types():
    products = benchmark._product_bands(2)
    spectra, band_weights = benchmark._nonuniform_host_inputs(4, 8, 8)
    starts = np.asarray((1, 5), dtype=np.int32)
    lengths = np.asarray((3, 2), dtype=np.int32)
    output, weights = benchmark._expected_results(
        spectra,
        band_weights,
        products,
        (1.0, 2.0),
        group_starts=starts,
        group_lengths=lengths,
        channels_to_average=1,
    )

    assert output.dtype == np.complex64
    assert weights.dtype == np.float64
    assert output.shape == (2, 2, 8)
    assert weights.shape == (2, 2)
    reference = np.sum(
        spectra[0][1:4] * np.conj(spectra[2][1:4]),
        axis=0,
        dtype=np.complex64,
    )
    np.testing.assert_allclose(output[0, 0], reference)
    np.testing.assert_allclose(
        weights[0, 0],
        np.sum(band_weights[0][1:4] * band_weights[2][1:4]),
    )
    assert np.unique(output).size > output.shape[-1]
    assert np.unique(weights).size == weights.size


def test_visibility_nonuniform_oracle_data_varies_by_band_block_and_channel():
    spectra, weights = benchmark._nonuniform_host_inputs(8, 6, 12)
    repeated_spectra, repeated_weights = benchmark._nonuniform_host_inputs(
        8, 6, 12
    )

    for first, repeated in zip(spectra, repeated_spectra, strict=True):
        np.testing.assert_array_equal(first, repeated)
        assert np.unique(first.real).size > first.shape[0]
        assert np.unique(first.imag).size > first.shape[1]
    for first, repeated in zip(weights, repeated_weights, strict=True):
        np.testing.assert_array_equal(first, repeated)
        assert np.unique(first).size == first.size
    assert all(
        not np.array_equal(spectra[0], spectrum) for spectrum in spectra[1:]
    )


def test_visibility_avg4_oracle_includes_channel_average_scale():
    spectra, weights = benchmark._nonuniform_host_inputs(2, 4, 8)
    output, _ = benchmark._expected_results(
        spectra,
        weights,
        ((0, 1),),
        (2.0,),
        group_starts=np.asarray((0,), dtype=np.int32),
        group_lengths=np.asarray((4,), dtype=np.int32),
        channels_to_average=4,
    )
    unaveraged = np.sum(
        spectra[0] * np.conj(spectra[1]), axis=0, dtype=np.complex64
    )
    reference = np.sum(
        unaveraged.reshape(2, 4), axis=1, dtype=np.complex64
    ) / np.float64(8.0)
    np.testing.assert_allclose(output[0, 0], reference.astype(np.complex64))


def test_visibility_timing_disables_range_validation():
    validation_flags = []

    def operation(*, validate_ranges):
        validation_flags.append(validate_ranges)

    benchmark._production_timing_operation(operation)()

    assert validation_flags == [False]


def test_visibility_benchmark_defaults_to_dual_pol_four_buffer_case():
    arguments = benchmark._parser().parse_args([])

    assert arguments.products == 4
    assert arguments.band_layout == "dual-pol"
    assert benchmark._product_bands(
        arguments.products, arguments.band_layout
    ) == ((0, 2), (0, 3), (1, 2), (1, 3))


@pytest.mark.parametrize(
    "values, message",
    [
        ((0, 8, 32, 1, 1, 0), "geometry values must be positive"),
        ((65_536, 8, 32, 1, 1, 0), "at most 65535 groups"),
        ((65_535, 65_535, 32, 1, 1, 0), "block count is out of range"),
        ((1, 8, 32, 2, 1, 0), "averaging must be one or four"),
        ((1, 8, 34, 4, 1, 0), "divisible by averaging"),
        ((1, 8, 32, 1, 0, 0), "iterations must be positive"),
    ],
)
def test_visibility_benchmark_rejects_invalid_geometry(values, message):
    with pytest.raises(ValueError, match=message):
        benchmark._validate_geometry(*values)


def test_benchmark_imports_kernel_factories_before_geometry_validation(
    monkeypatch,
):
    import sys
    from types import SimpleNamespace

    # No device is needed to exercise the real runner's lazy imports.
    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace())
    arguments = benchmark._parser().parse_args(["--groups", "0"])
    with pytest.raises(ValueError, match="geometry values must be positive"):
        benchmark._run(arguments)
