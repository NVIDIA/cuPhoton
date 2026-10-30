# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Host stages for the opt-in structured detector fit.

The supported fit has 798 samples, 30 requested components, 200 singular
values and a degree-597 prediction polynomial. Selected roots and their
conjugates require double-double refinement before the original P2 solve.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from time import perf_counter

import numpy as np

from . import linear_prediction as lp

BACKEND_VERSION = "structured-rds-v1"
REFINEMENT_VERSION = "selected-dd-retained-decay-v1"


@dataclass(frozen=True)
class PreparedStructuredFit:
    """One host P1 result retained until its root batch completes."""

    time: np.ndarray
    trace: np.ndarray
    coefficients: np.ndarray
    singular_values: np.ndarray
    selected_model_order: int


@dataclass(frozen=True)
class _RootSelection:
    active_sequence: tuple[int, ...]
    classifications: tuple[tuple[int, bool, int, bool], ...]
    zero_count: int
    decaying_root_count: int
    selected_indices: tuple[int, ...]
    modes: lp.LinearPredictionModes = field(compare=False)


@dataclass(frozen=True)
class _RefinedRoots:
    roots: np.ndarray
    indices: np.ndarray
    selected_modes: tuple[int, ...]
    selected_roles: tuple[bool, ...]
    seeds: np.ndarray
    outputs: np.ndarray
    history: np.ndarray
    status_steps: np.ndarray
    selection: _RootSelection


def _require(condition: bool | np.bool_, message: str) -> None:
    if not condition:
        raise ValueError(message)


def prepare_fit(
    time, trace, *, n_components: int = 30
) -> PreparedStructuredFit:
    """Run the shared P1 stage for one supported detector trace."""
    _require(n_components == 30, "structured fits require 30 components")
    bins, signal = lp._validate_inputs(np, time, trace, n_components)
    _require(bins.shape == (798,), "structured fits require 798 samples")
    _require(
        bool(np.isfinite(bins).all() and np.isfinite(signal).all()),
        "structured fit inputs must be finite",
    )
    _require(bins[1] > bins[0], "structured fits require increasing time")
    p1 = lp._linear_prediction_p1_coefficients_impl(
        xp=np, time=bins, trace=signal, n_components=n_components
    )
    _require(
        p1.coefficients.shape == (597,)
        and p1.singular_values.shape == (200,)
        and bool(np.isfinite(p1.coefficients).all())
        and bool(np.isfinite(p1.singular_values).all()),
        "structured P1 produced nonfinite or unsupported output",
    )
    return PreparedStructuredFit(
        time=bins,
        trace=signal,
        coefficients=p1.coefficients,
        singular_values=p1.singular_values,
        selected_model_order=p1.selected_model_order,
    )


def prediction_polynomials(
    rows: Sequence[PreparedStructuredFit],
) -> np.ndarray:
    """Return the monic polynomials in descending coefficient order."""
    if not rows:
        return np.empty((0, 598), dtype=np.float64)
    coefficients = np.stack([row.coefficients for row in rows])
    return np.ascontiguousarray(
        np.column_stack((np.ones(len(rows)), -coefficients))
    )


def _tracked_selection(roots: np.ndarray, time: np.ndarray) -> _RootSelection:
    """Track root identities through the shared production selection."""
    selection = lp._linear_prediction_root_selection(
        np, time=time, roots=roots, singular_value_count=200
    )
    return _RootSelection(
        active_sequence=tuple(int(i) for i in selection.active_indices),
        classifications=tuple(
            (int(i), bool(d >= 0), int(np.sign(w)), bool(w == 0))
            for i, d, w in zip(
                selection.active_indices,
                selection.active_decay,
                selection.active_angular_frequency,
                strict=True,
            )
        ),
        zero_count=selection.zero_count,
        decaying_root_count=selection.modes.decaying_root_count,
        selected_indices=tuple(int(i) for i in selection.selected_indices),
        modes=selection.modes,
    )


def _same_selection(before: _RootSelection, after: _RootSelection) -> bool:
    # Refinement can break an exact decay tie without changing the modes.
    # Preserve the root set and every classification; only an original exact
    # tie may change order, following the refined production ordering.
    original_decay = dict(
        zip(before.selected_indices, before.modes.decay, strict=True)
    )
    return (
        before.active_sequence == after.active_sequence
        and before.classifications == after.classifications
        and before.zero_count == after.zero_count
        and before.decaying_root_count == after.decaying_root_count
        and set(before.selected_indices) == set(after.selected_indices)
        and all(
            original_decay[old] == original_decay[new]
            for old, new in zip(
                before.selected_indices, after.selected_indices, strict=True
            )
        )
    )


def _refine_selected_roots(
    polynomial: np.ndarray, roots: np.ndarray, time: np.ndarray
) -> _RefinedRoots:
    from . import _structured_native

    _require(
        roots.shape == (597,) and bool(np.isfinite(roots).all()),
        "structured root solve must return 597 finite roots",
    )
    policy = _tracked_selection(roots, time)
    indices_list: list[int] = []
    selected_modes: list[int] = []
    selected_roles: list[bool] = []
    pairs: list[tuple[int, int]] = []
    separations: list[float] = []
    for mode, index in enumerate(policy.selected_indices):
        seed = roots[index]
        targets = [(index, True)]
        if seed.imag != 0:
            partners = np.flatnonzero(roots == np.conj(seed))
            _require(
                len(partners) == 1 and partners[0] != index,
                "missing or ambiguous exact conjugate partner",
            )
            partner = int(partners[0])
            pairs.append((index, partner))
            targets.append((partner, False))
        for target, selected in targets:
            separation = float(
                np.min(np.abs(np.delete(roots, target) - roots[target]))
            )
            _require(
                math.isfinite(separation) and separation > 0,
                "duplicate or ambiguous refinement seed",
            )
            indices_list.append(target)
            selected_modes.append(mode)
            selected_roles.append(selected)
            separations.append(separation)
    _require(
        len(set(indices_list)) == len(indices_list),
        "selected and partner root identities overlap",
    )
    indices = np.asarray(indices_list, dtype=np.int32)
    seeds = np.asarray(
        [[roots[i].real, roots[i].imag] for i in indices], dtype=np.float64
    ).reshape(-1, 2)
    seed_bytes = seeds.tobytes()
    separation_array = np.asarray(separations, dtype=np.float64)
    values = seeds.copy()
    history = np.full((len(indices), 8, 8), np.nan, dtype=np.float64)
    statuses = np.full((len(indices), 2), -1, dtype=np.int32)
    for start in range(0, len(indices), 32):
        stop = min(start + 32, len(indices))
        outputs, steps, status = _structured_native.refine_selected(
            polynomial, seeds[start:stop], separation_array[start:stop]
        )
        values[start:stop] = outputs
        history[start:stop] = steps
        statuses[start:stop] = status
    failures = np.flatnonzero(statuses[:, 0] != 0)
    if len(failures):
        failed = int(failures[0])
        raise ValueError(
            "selected-root double-double refinement failed: "
            f"native status {statuses[failed, 0]} at root {indices[failed]} "
            f"after {statuses[failed, 1]} iterations"
        )
    _require(
        seeds.tobytes() == seed_bytes,
        "double-double refinement modified its input seeds",
    )
    candidate = roots.copy()
    candidate[indices] = values[:, 0] + 1j * values[:, 1]
    _require(
        bool(np.isfinite(candidate).all()),
        "selected-root double-double refinement returned nonfinite roots",
    )
    _require(
        all(candidate[b] == np.conj(candidate[a]) for a, b in pairs),
        "double-double refinement changed conjugacy",
    )
    refined_policy = _tracked_selection(candidate, time)
    _require(
        _same_selection(policy, refined_policy),
        "double-double refinement changed original mode selection",
    )
    mode_positions = {
        index: mode
        for mode, index in enumerate(refined_policy.selected_indices)
    }
    return _RefinedRoots(
        roots=candidate,
        indices=indices,
        selected_modes=tuple(
            mode_positions[policy.selected_indices[mode]]
            for mode in selected_modes
        ),
        selected_roles=tuple(selected_roles),
        seeds=seeds,
        outputs=values,
        history=history,
        status_steps=statuses,
        selection=refined_policy,
    )


def _retained_modes(
    refined: _RefinedRoots, time: np.ndarray
) -> lp.LinearPredictionModes:
    """Retain accepted DD low words for eligible positive real-mode decays."""
    roots = refined.roots
    policy = _tracked_selection(roots, time)
    _require(
        policy == refined.selection,
        "production layout changes accepted DD selection",
    )
    baseline = policy.modes
    delta_t = float(time[1] - time[0])
    _require(
        math.isfinite(delta_t) and delta_t > 0,
        "invalid original time increment",
    )
    changed = baseline.decay.copy()
    for mode, index in enumerate(policy.selected_indices):
        z = roots[index]
        if z.imag != 0 or z.real <= 0:
            continue
        _require(
            baseline.angular_frequency[mode] == 0,
            "unexpected positive real-mode frequency",
        )
        positions = np.flatnonzero(refined.indices == index)
        _require(
            len(positions) == 1, "missing or ambiguous selected DD entry"
        )
        slot = int(positions[0])
        step = int(refined.status_steps[slot, 1])
        _require(
            1 <= step <= 8
            and refined.selected_roles[slot]
            and refined.selected_modes[slot] == mode,
            "accepted DD slot or selected role differs",
        )
        words = refined.history[slot, step - 1]
        hi, lo, im_hi, im_lo = map(float, words[:4])
        _require(
            bool(np.isfinite(words).all())
            and im_hi == im_lo == 0
            and words[7] >= 2
            and 0 <= words[4] <= 2.0**-80
            and 0 <= words[5] <= 2.0**-80,
            "real DD words fail accepted convergence conditions",
        )
        rounded = np.asarray([hi + lo, im_hi + im_lo], dtype=np.float64)
        _require(
            rounded.tobytes()
            == refined.outputs[slot].tobytes()
            == np.asarray([z.real, z.imag], dtype=np.float64).tobytes(),
            "accepted DD rounding differs from refined root",
        )
        if not 0.5 <= hi <= 2:
            continue
        # Sterbenz interval: preserve the low word when subtracting one.
        offset = math.fsum([hi, -1.0, lo])
        _require(offset > -1, "retained real root lies outside log1p domain")
        new_decay = math.log1p(offset) / delta_t
        _require(
            math.isfinite(new_decay)
            and (new_decay >= 0) == (baseline.decay[mode] >= 0),
            "retained decay crosses selection boundary",
        )
        changed[mode] = new_decay
    _require(
        bool(np.isfinite(changed).all())
        and np.array_equal(changed >= 0, baseline.decay >= 0)
        and np.array_equal((-changed).argsort(), (-baseline.decay).argsort()),
        "retained decay changes original mode order",
    )
    return replace(baseline, decay=changed)


def finish_fit(
    prepared: PreparedStructuredFit,
    roots,
    *,
    root_device: str = "cpu",
    fit_diagnostics: bool = False,
) -> lp.LinearPredictionResult:
    """Refine a root row and execute the shared original P2/result stage."""
    _require(root_device in {"cpu", "gpu"}, "invalid structured root device")
    _require(
        prepared.time.shape == prepared.trace.shape == (798,)
        and prepared.coefficients.shape == (597,)
        and prepared.singular_values.shape == (200,)
        and 0 < prepared.selected_model_order <= 30,
        "unsupported prepared structured fit",
    )
    started = perf_counter()
    polynomial = prediction_polynomials([prepared])[0]
    refined = _refine_selected_roots(
        polynomial, np.asarray(roots, dtype=np.complex128), prepared.time
    )
    modes = _retained_modes(refined, prepared.time)
    result = lp._linear_prediction_result_from_modes(
        xp=np,
        backend=f"structured-{root_device}",
        time=prepared.time,
        trace=prepared.trace,
        modes=modes,
        singular_values=prepared.singular_values,
        selected_model_order=prepared.selected_model_order,
        roots_stats=lp.PredictionRootsStats(
            backend=BACKEND_VERSION,
            array_module="numpy" if root_device == "cpu" else "cupy",
            matrix_size=597,
            row_count=1,
            failures=0,
            elapsed_s=0.0,
        ),
        p2_ridge_alpha=0.0,
        fit_diagnostics=fit_diagnostics,
    )
    return replace(result, elapsed_s=perf_counter() - started)
