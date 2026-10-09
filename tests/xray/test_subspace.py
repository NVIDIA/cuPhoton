# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import builtins

import numpy as np
import pytest

from cuphoton.xray.linear_prediction import synthetic_trace
from cuphoton.xray.subspace import (
    compare_subspace_methods,
    compare_subspace_methods_batch,
    esprit_roots,
    matrix_pencil_roots,
)


def test_matrix_pencil_and_esprit_return_model_order_roots():
    _time, trace = synthetic_trace(samples=64)

    pencil_roots = matrix_pencil_roots(trace, model_order=5)
    esprit_values = esprit_roots(trace, model_order=5)
    randomized_roots = matrix_pencil_roots(
        trace,
        model_order=5,
        svd_backend="randomized",
    )

    assert pencil_roots.shape == (5,)
    assert esprit_values.shape == (5,)
    assert randomized_roots.shape == (5,)
    assert np.all(np.isfinite(pencil_roots))
    assert np.all(np.isfinite(esprit_values))
    assert np.all(np.isfinite(randomized_roots))


def test_compare_subspace_methods_reports_residuals():
    time, trace = synthetic_trace(samples=64)

    result = compare_subspace_methods(
        time,
        trace,
        model_order=5,
        components=6,
        methods=("matrix-pencil", "esprit"),
    )

    assert result.samples == 64
    assert result.model_order == 5
    assert result.components == 6
    assert result.baseline_chi2 >= 0
    assert result.baseline_rms_residual >= 0
    assert [method.method for method in result.methods] == [
        "matrix-pencil",
        "esprit",
    ]
    for method in result.methods:
        assert method.samples == 64
        assert method.model_order == 5
        assert method.pencil_rows == 32
        assert method.svd_backend == "full"
        assert 0 < method.svd_rank <= method.model_order
        assert len(method.singular_value_head) > 0
        assert method.rms_residual >= 0
        assert method.max_abs_reconstruction_diff >= 0
        assert method.elapsed_s > 0
        assert len(method.roots_real) == 5
        assert len(method.roots_imag) == 5


def test_compare_subspace_methods_can_use_randomized_truncated_svd():
    time, trace = synthetic_trace(samples=64)

    result = compare_subspace_methods(
        time,
        trace,
        model_order=5,
        components=6,
        methods=("matrix-pencil",),
        svd_backends=("randomized",),
        randomized_oversamples=4,
        power_iterations=1,
        random_seed=7,
    )

    assert len(result.methods) == 1
    method = result.methods[0]
    assert method.method == "matrix-pencil"
    assert method.svd_backend == "randomized"
    assert method.svd_rank == 5
    assert len(method.singular_value_head) == 5
    assert method.rms_residual >= 0
    assert len(method.roots_real) == 5


def test_compare_subspace_methods_can_use_partial_svd():
    pytest.importorskip("scipy.sparse.linalg")

    time, trace = synthetic_trace(samples=64)

    result = compare_subspace_methods(
        time,
        trace,
        model_order=5,
        components=6,
        methods=("matrix-pencil",),
        svd_backends=("partial",),
        random_seed=7,
    )

    assert len(result.methods) == 1
    method = result.methods[0]
    assert method.method == "matrix-pencil"
    assert method.svd_backend == "partial"
    assert method.svd_rank == 5
    assert len(method.singular_value_head) == 5
    assert method.rms_residual >= 0
    assert len(method.roots_real) == 5


def test_subspace_batch_preserves_trace_results_and_source_order():
    time, trace = synthetic_trace(samples=64)
    trace_rows = np.stack((trace, trace + 0.1 * np.cos(3.0 * time)))
    sources = (
        {"kind": "trace-npz", "path": "trace-y2.npz"},
        {
            "kind": "trace-npz",
            "path": "trace-y10.npz",
            "summary": {"detector_row": 10},
            "time_matches_first": True,
        },
    )
    options = {
        "model_order": 4,
        "components": 6,
        "methods": ("esprit", "matrix-pencil"),
        "svd_backends": ("randomized", "full"),
        "pencil_rows": 24,
        "randomized_oversamples": 3,
        "power_iterations": 2,
        "random_seed": 9,
    }
    batch = compare_subspace_methods_batch(
        time, trace_rows, sources, **options
    )
    payload = batch.to_dict()

    assert payload["source"]["traces"] == sources
    assert payload["trace_count"] == 2
    for index, (trace_result, trace_payload) in enumerate(
        zip(batch.traces, payload["traces"], strict=True)
    ):
        expected = compare_subspace_methods(
            time, trace_rows[index], **options
        )
        assert trace_payload["trace_index"] == index
        assert trace_payload["source"] == sources[index]
        assert trace_payload["svd_backends"] == options["svd_backends"]
        assert trace_result.baseline_chi2 == expected.baseline_chi2
        for result, expected_method in zip(
            trace_result.methods, expected.methods, strict=True
        ):
            assert result.pencil_rows == 24
            assert result.rms_residual == pytest.approx(
                expected_method.rms_residual
            )
            np.testing.assert_allclose(
                result.roots_real, expected_method.roots_real
            )
            np.testing.assert_allclose(
                result.roots_imag, expected_method.roots_imag
            )

    assert [
        (summary.method, summary.svd_backend)
        for summary in batch.method_summary
    ] == [
        ("esprit", "randomized"),
        ("esprit", "full"),
        ("matrix-pencil", "randomized"),
        ("matrix-pencil", "full"),
    ]
    for summary in batch.method_summary:
        methods = [
            method
            for result in batch.traces
            for method in result.methods
            if (method.method, method.svd_backend)
            == (summary.method, summary.svd_backend)
        ]
        assert summary.trace_count == 2
        assert summary.rms_residual_min == min(
            method.rms_residual for method in methods
        )
        assert summary.rms_residual_max == max(
            method.rms_residual for method in methods
        )
        assert summary.max_abs_reconstruction_diff_max == max(
            method.max_abs_reconstruction_diff for method in methods
        )
        assert summary.elapsed_s_total == pytest.approx(
            sum(method.elapsed_s for method in methods)
        )

    assert payload["baseline_chi2_min"] == min(
        result.baseline_chi2 for result in batch.traces
    )
    assert payload["baseline_chi2_max"] == max(
        result.baseline_chi2 for result in batch.traces
    )
    assert payload["baseline_rms_residual_min"] == min(
        result.baseline_rms_residual for result in batch.traces
    )
    assert payload["baseline_rms_residual_max"] == max(
        result.baseline_rms_residual for result in batch.traces
    )


def test_matrix_pencil_rejects_rankless_trace():
    with pytest.raises(ValueError, match="rank"):
        matrix_pencil_roots(np.zeros(32), model_order=5)


def test_partial_svd_reports_solver_failure():
    pytest.importorskip("scipy.sparse.linalg")

    with pytest.raises(ValueError, match="partial SVD failed"):
        matrix_pencil_roots(
            np.zeros(32),
            model_order=5,
            svd_backend="partial",
        )


def test_partial_svd_reports_missing_scipy(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "scipy.sparse.linalg":
            raise ImportError("missing scipy")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ValueError, match="partial SVD requires scipy"):
        matrix_pencil_roots(
            np.ones(32),
            model_order=5,
            svd_backend="partial",
        )


@pytest.mark.parametrize("model_order", [0, -1])
def test_subspace_helpers_reject_non_positive_model_order(model_order):
    _time, trace = synthetic_trace(samples=32)

    with pytest.raises(ValueError, match="model_order must be positive"):
        matrix_pencil_roots(trace, model_order=model_order)
    with pytest.raises(ValueError, match="model_order must be positive"):
        esprit_roots(trace, model_order=model_order)
