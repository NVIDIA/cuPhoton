# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen operating points and field-based uncertainty for Model Alerts."""

from __future__ import annotations

import numpy as np


def checked(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or labels.shape != scores.shape or not len(labels):
        raise ValueError(
            "labels and scores must be nonempty matching vectors"
        )
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("binary labels and finite scores are required")
    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("scores must lie in [0, 1]")
    return labels.astype(bool), scores


def recall_threshold(labels, scores, recall=0.95):
    """Highest inclusive threshold satisfying the requested validation recall.

    The kth positive order statistic retains all tied scores. Negative scores
    cannot increase this threshold without losing the required positives.
    """
    labels, scores = checked(labels, scores)
    if not 0 < recall <= 1 or not labels.any():
        raise ValueError("recall must be in (0, 1] and positives must exist")
    positive = np.sort(scores[labels])[::-1]
    return float(positive[int(np.ceil(recall * len(positive))) - 1])


def confusion(labels, scores, threshold):
    labels, scores = checked(labels, scores)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")
    retained = scores >= threshold
    tp = int(np.sum(labels & retained))
    fp = int(np.sum(~labels & retained))
    fn = int(np.sum(labels & ~retained))
    tn = int(np.sum(~labels & ~retained))
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        recall=tp / (tp + fn) if tp + fn else None,
        fpr=fp / (fp + tn) if fp + tn else None,
        brier=float(np.mean((scores - labels) ** 2)),
    )


def queue_projection(tpr, fpr, prevalence, count):
    """Expected queue under fixed class-conditional score distributions."""
    values = np.asarray([tpr, fpr, prevalence], dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("rates and prevalence must lie in [0, 1]")
    if not np.isfinite(count) or count < 0:
        raise ValueError("population count must be finite and nonnegative")
    real, bogus = prevalence * tpr, (1 - prevalence) * fpr
    return dict(
        queue=count * (real + bogus),
        completeness=tpr,
        purity=real / (real + bogus) if real + bogus else None,
    )


def paired_field_bootstrap(
    labels,
    scores_a,
    scores_b,
    fields,
    threshold_a,
    threshold_b,
    *,
    seed=731,
    draws=2000,
):
    """Field intervals for FPR and paired A minus B at frozen thresholds.

    Sample fields with replacement, preserving every candidate and epoch in
    each selected field. Replicates lacking negatives have undefined FPR and
    are counted explicitly rather than replaced by zero.
    """
    labels, scores_a = checked(labels, scores_a)
    _, scores_b = checked(labels, scores_b)
    fields = np.asarray(fields)
    if fields.shape != labels.shape or draws < 1:
        raise ValueError(
            "matching field identifiers and positive draws required"
        )
    unique, inverse = np.unique(fields, return_inverse=True)
    negatives = np.bincount(inverse, weights=~labels)
    false_a = np.bincount(
        inverse, weights=(~labels & (scores_a >= threshold_a))
    )
    false_b = np.bincount(
        inverse, weights=(~labels & (scores_b >= threshold_b))
    )
    rng = np.random.default_rng(seed)
    outcomes = []
    for _ in range(draws):
        selected = rng.integers(len(unique), size=len(unique))
        denominator = negatives[selected].sum()
        if denominator:
            a = false_a[selected].sum() / denominator
            b = false_b[selected].sum() / denominator
            outcomes.append((a, b, a - b))
    if not outcomes:
        raise ValueError("no bootstrap replicate contains negatives")
    intervals = np.quantile(outcomes, [0.025, 0.975], axis=0).T
    return dict(
        zip(("a", "b", "a_minus_b"), intervals.tolist(), strict=False)
    ) | {
        "draws": draws,
        "undefined_draws": draws - len(outcomes),
        "fields": len(unique),
        "seed": seed,
    }


def field_rate_intervals(
    labels, scores, fields, threshold, *, seed=731, draws=2000
):
    """Field intervals for recall/FPR at a frozen inclusive threshold."""
    labels, scores = checked(labels, scores)
    fields = np.asarray(fields)
    if fields.shape != labels.shape or draws < 1:
        raise ValueError(
            "matching field IDs and positive draw count required"
        )
    _, inverse = np.unique(fields, return_inverse=True)
    counts = np.array(
        [
            np.bincount(inverse, weights=values)
            for values in (
                labels,
                ~labels,
                labels & (scores >= threshold),
                ~labels & (scores >= threshold),
            )
        ]
    )
    rng = np.random.default_rng(seed)
    rates = {"recall": [], "fpr": []}
    for _ in range(draws):
        sample = rng.integers(counts.shape[1], size=counts.shape[1])
        positive, negative, tp, fp = counts[:, sample].sum(axis=1)
        if positive:
            rates["recall"].append(tp / positive)
        if negative:
            rates["fpr"].append(fp / negative)
    return {
        name: dict(
            interval=np.quantile(values, [0.025, 0.975]).tolist()
            if values
            else None,
            undefined_draws=draws - len(values),
            draws=draws,
            seed=seed,
        )
        for name, values in rates.items()
    }
