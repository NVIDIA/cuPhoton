# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from cuphoton.xray import linear_prediction as lp
from cuphoton.xray import structured


def _roots():
    roots = np.linspace(0.1, 0.7, 597).astype(np.complex128)
    roots[-3] = 1.0
    roots[-2] = 1.05 * np.exp(-0.1j)
    roots[-1] = np.conj(roots[-2])
    return roots


def _prepared():
    return structured.PreparedStructuredFit(
        time=np.arange(798, dtype=np.float64),
        trace=np.ones(798),
        coefficients=np.ones(597),
        singular_values=np.ones(200),
        selected_model_order=30,
    )


def _tied_roots():
    roots = _roots()
    roots[-5:] = [1, 1.5 - 2j, 1.5 + 2j, 2 - 1.5j, 2 + 1.5j]
    return roots


def _refiner(monkeypatch, *, real_low=0.0, status=0, alter=None):
    from cuphoton.xray import _structured_native

    def refine(polynomial, seeds, separation):
        assert polynomial.shape == (598,)
        assert np.all(separation > 0)
        values = seeds.copy()
        history = np.full((len(seeds), 8, 8), np.nan)
        history[:, 1] = 0
        history[:, 1, 0] = seeds[:, 0]
        history[:, 1, 1] = np.where(seeds[:, 1] == 0, real_low, 0)
        history[:, 1, 2] = seeds[:, 1]
        history[:, 1, 7] = 2
        statuses = np.tile(
            np.array([status, 2], dtype=np.int32), (len(seeds), 1)
        )
        if alter is not None:
            alter(values, history, statuses)
        return values, history, statuses

    monkeypatch.setattr(_structured_native, "refine_selected", refine)


def _refined():
    return structured._refine_selected_roots(
        np.ones(598), _roots(), np.arange(798, dtype=np.float64)
    )


@pytest.mark.parametrize(
    ("samples", "components", "message"),
    [(798, 29, "30 components"), (797, 30, "798 samples")],
)
def test_prepare_rejects_unsupported_fit_before_svd(
    monkeypatch, samples, components, message
):
    def unexpected(*args, **kwargs):
        raise AssertionError("unsupported fits must not reach P1")

    monkeypatch.setattr(
        lp, "_linear_prediction_p1_coefficients_impl", unexpected
    )
    with pytest.raises(ValueError, match=message):
        structured.prepare_fit(
            np.arange(samples), np.ones(samples), n_components=components
        )


def test_prepare_preserves_shared_p1_coefficients(monkeypatch):
    coefficients = np.linspace(-1, 1, 597)
    singular_values = np.linspace(200, 1, 200)

    def p1(*, xp, time, trace, n_components):
        assert xp is np
        assert n_components == 30
        return lp._P1CoefficientsResult(singular_values, coefficients, 27)

    monkeypatch.setattr(lp, "_linear_prediction_p1_coefficients_impl", p1)
    prepared = structured.prepare_fit(np.arange(798), np.ones(798))
    polynomial = structured.prediction_polynomials([prepared])
    assert prepared.selected_model_order == 27
    assert prepared.coefficients is coefficients
    assert prepared.singular_values is singular_values
    assert polynomial.flags.c_contiguous
    np.testing.assert_array_equal(polynomial[0], np.r_[1.0, -coefficients])


def test_tracked_selection_uses_production_negative_stride_view(monkeypatch):
    original_abs = np.abs
    seen = []

    def checked_abs(values):
        if np.iscomplexobj(values):
            assert values.strides[0] < 0
            seen.append(values.copy())
        return original_abs(values)

    monkeypatch.setattr(np, "abs", checked_abs)
    roots = _roots()
    times = np.arange(798, dtype=np.float64)
    tracked = structured._tracked_selection(roots, times)
    modes = lp._linear_prediction_modes_from_roots(
        np, time=times, roots=roots, singular_value_count=200
    )
    assert len(seen) == 2
    np.testing.assert_array_equal(seen[0], np.sort(roots)[::-1][:200])
    assert tracked.selected_indices == (595, 594)
    assert tracked.decaying_root_count == modes.decaying_root_count == 3
    np.testing.assert_array_equal(tracked.modes.decay, modes.decay)
    np.testing.assert_array_equal(
        tracked.modes.angular_frequency, modes.angular_frequency
    )


def test_finish_fit_uses_moved_refined_root(monkeypatch):
    moved = np.nextafter(1.0, np.inf)

    def alter(values, history, statuses):
        real = values[:, 1] == 0
        values[real, 0] = moved
        history[real, 1, 0] = moved

    _refiner(monkeypatch, alter=alter)
    prepared = _prepared()
    roots = _roots()
    result = structured.finish_fit(prepared, roots)
    assert roots[-3] == 1.0
    assert result.decay[-1] == math.log1p(moved - 1.0) > 0
    assert result.angular_frequency[-1] == 0
    assert np.isfinite(result.reconstruction).all()


def test_refinement_breaks_exact_decay_tie_in_production_order(monkeypatch):
    def alter(values, history, statuses):
        second_pair = values[:, 0] == 2.0
        values[second_pair, 0] = np.nextafter(2.0, np.inf)

    _refiner(monkeypatch, alter=alter, real_low=2.0**-60)
    roots = _tied_roots()
    times = np.arange(798, dtype=np.float64)
    before = structured._tracked_selection(roots, times)
    assert before.modes.decay[0] == before.modes.decay[1]
    assert before.selected_indices[:2] == (593, 595)
    refined = structured._refine_selected_roots(np.ones(598), roots, times)
    assert refined.selection.selected_indices[:2] == (595, 593)
    modes = structured._retained_modes(refined, times)
    expected = lp._linear_prediction_modes_from_roots(
        np, time=times, roots=refined.roots, singular_value_count=200
    )
    np.testing.assert_array_equal(modes.decay[:2], expected.decay[:2])
    np.testing.assert_array_equal(
        modes.angular_frequency, expected.angular_frequency
    )
    assert modes.decay[-1] == math.log1p(2.0**-60)
    for slot, index in enumerate(refined.indices):
        if refined.selected_roles[slot]:
            mode = refined.selected_modes[slot]
            assert refined.selection.selected_indices[mode] == index


def test_refinement_rejects_reordering_unequal_decays(monkeypatch):
    def alter(values, history, statuses):
        first_pair = values[:, 0] == 1.5
        values[first_pair, 0] += 1e-12

    _refiner(monkeypatch, alter=alter)
    roots = _tied_roots()
    roots[-2:] = np.nextafter(2.0, np.inf) + 1j * roots[-2:].imag
    times = np.arange(798, dtype=np.float64)
    before = structured._tracked_selection(roots, times)
    assert before.modes.decay[0] > before.modes.decay[1]
    with pytest.raises(ValueError, match="original mode selection"):
        structured._refine_selected_roots(np.ones(598), roots, times)


def test_refinement_preserves_complex_modes_and_retains_real_low_word(
    monkeypatch,
):
    low = 2.0**-60
    _refiner(monkeypatch, real_low=low)
    refined = _refined()
    times = np.arange(798, dtype=np.float64)
    modes = structured._retained_modes(refined, times)
    original = lp._linear_prediction_modes_from_roots(
        np, time=times, roots=_roots(), singular_value_count=200
    )
    np.testing.assert_array_equal(
        modes.angular_frequency, original.angular_frequency
    )
    assert modes.decaying_root_count == original.decaying_root_count
    assert modes.decay[0] == original.decay[0]
    assert original.decay[-1] == 0
    assert modes.decay[-1] == math.log1p(low)


@pytest.mark.parametrize("status", [1, 2, 3, 4, 5, 6])
def test_refinement_rejections_are_not_silently_used(monkeypatch, status):
    _refiner(monkeypatch, status=status)
    with pytest.raises(ValueError, match=f"native status {status} at root"):
        _refined()


def test_refinement_rejects_missing_exact_conjugate(monkeypatch):
    _refiner(monkeypatch)
    roots = _roots()
    roots[-1] += 1e-10j
    with pytest.raises(ValueError, match="conjugate partner"):
        structured._refine_selected_roots(
            np.ones(598), roots, np.arange(798, dtype=np.float64)
        )


def test_refinement_rejects_changed_selection(monkeypatch):
    def alter(values, history, statuses):
        values[values[:, 1] == 0, 0] = 0.999

    _refiner(monkeypatch, alter=alter)
    with pytest.raises(ValueError, match="original mode selection"):
        _refined()


def test_finish_rejects_zero_decay_crossing_after_low_word_retention(
    monkeypatch,
):
    _refiner(monkeypatch, real_low=-(2.0**-60))
    with pytest.raises(ValueError, match="crosses selection boundary"):
        structured.finish_fit(_prepared(), _roots())


def test_retained_low_word_cannot_reorder_tied_modes(monkeypatch):
    _refiner(monkeypatch, real_low=2.0**-60)
    roots = _roots()
    roots[-2:] = [0.6 - 0.8j, 0.6 + 0.8j]
    times = np.arange(798, dtype=np.float64)
    refined = structured._refine_selected_roots(np.ones(598), roots, times)
    np.testing.assert_array_equal(refined.selection.modes.decay, [0, 0])
    with pytest.raises(
        ValueError, match="retained decay changes original mode order"
    ):
        structured._retained_modes(refined, times)


def test_retained_decay_rejects_unconverged_words(monkeypatch):
    def alter(values, history, statuses):
        history[:, 1, 4] = 2.0**-79

    _refiner(monkeypatch, alter=alter)
    with pytest.raises(ValueError, match="convergence conditions"):
        structured._retained_modes(
            _refined(), np.arange(798, dtype=np.float64)
        )


def test_retained_decay_rejects_wrong_root_slot_order(monkeypatch):
    _refiner(monkeypatch)
    refined = _refined()
    refined = replace(
        refined,
        selection=replace(
            refined.selection,
            selected_indices=refined.selection.selected_indices[::-1],
        ),
    )
    with pytest.raises(ValueError, match="production layout"):
        structured._retained_modes(refined, np.arange(798, dtype=np.float64))


def test_retained_decay_rejects_wrong_dd_mode_association(monkeypatch):
    _refiner(monkeypatch)
    refined = _refined()
    refined = replace(refined, selected_modes=refined.selected_modes[::-1])
    with pytest.raises(ValueError, match="accepted DD slot"):
        structured._retained_modes(refined, np.arange(798, dtype=np.float64))


def test_retained_decay_updates_multiple_eligible_real_modes(monkeypatch):
    low = 2.0**-60
    _refiner(monkeypatch, real_low=low)
    roots = _roots()
    roots[-4] = 1.0 + np.spacing(1.0)
    times = np.arange(798, dtype=np.float64)
    refined = structured._refine_selected_roots(np.ones(598), roots, times)
    modes = structured._retained_modes(refined, times)
    np.testing.assert_array_equal(
        modes.decay[modes.angular_frequency == 0],
        [math.log1p(np.spacing(1.0) + low), math.log1p(low)],
    )


def test_no_selected_modes_still_returns_original_zero_mode_fit(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("there are no selected roots to refine")

    from cuphoton.xray import _structured_native

    monkeypatch.setattr(_structured_native, "refine_selected", unexpected)
    prepared = _prepared()
    roots = np.linspace(0.1, 0.7, 597).astype(np.complex128)
    result = structured.finish_fit(prepared, roots, fit_diagnostics=True)
    assert result.decay.shape == (0,)
    assert result.time_components.shape == (798, 0)
    assert result.spectrum_components.shape == (1000, 0)
    np.testing.assert_allclose(
        result.reconstruction, prepared.trace, atol=1e-14
    )
    assert result.diagnostics is not None
    assert result.diagnostics.p2_rank == 1
    assert result.roots_stats is not None
    assert result.roots_stats.backend == structured.BACKEND_VERSION
