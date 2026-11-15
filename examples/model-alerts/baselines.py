# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Measured-image controls; no injection metadata enters the features."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

FEATURE_NAMES = (
    "measured_detection_snr",
    "positive_core_fraction",
    "negative_absolute_flux_fraction",
    "difference_peak_to_rms",
    "search_template_correlation",
    "nonfinite_fraction",
    "repeated_extreme_fraction",
)


def measured_features(images, snr):
    """Simple statistics from complete, unclipped raw electron cutouts.

    Measure nonfinite pixels and repeated extrema rather than assuming they
    are absent. Tied extrema can indicate clipping or quantization; they do
    not prove either. Missing pixels are zero-filled only for this diagnostic
    calculation; generated model inputs are required to be finite.
    """
    images = np.asarray(images, dtype=float)
    snr = np.asarray(snr, dtype=float)
    if (
        images.ndim != 4
        or images.shape[1] != 3
        or snr.shape != (len(images),)
    ):
        raise ValueError("candidate triplets and one measured SNR required")
    nonfinite = (~np.isfinite(images)).mean(axis=(1, 2, 3))
    images = np.where(np.isfinite(images), images, 0)
    planes = images.reshape(len(images), 3, -1)
    low = (planes == planes.min(axis=2, keepdims=True)).sum(axis=2)
    high = (planes == planes.max(axis=2, keepdims=True)).sum(axis=2)
    repeated = np.minimum(
        planes.shape[2], np.maximum(low - 1, 0) + np.maximum(high - 1, 0)
    ).sum(axis=1) / (3 * planes.shape[2])
    search, reference, difference = images[:, 0], images[:, 1], images[:, 2]
    positive = np.maximum(difference, 0)
    absolute = np.abs(difference).sum(axis=(1, 2)) + 1e-12
    center = difference.shape[-1] // 2
    core = positive[:, center - 2 : center + 3, center - 2 : center + 3].sum(
        axis=(1, 2)
    )
    rms = np.sqrt(np.mean(difference**2, axis=(1, 2))) + 1e-12
    a = search.reshape(len(search), -1)
    b = reference.reshape(len(reference), -1)
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    correlation = (a * b).sum(axis=1) / np.sqrt(
        np.maximum((a * a).sum(axis=1) * (b * b).sum(axis=1), 1e-24)
    )
    return np.column_stack(
        (
            snr,
            core / absolute,
            np.maximum(-difference, 0).sum(axis=(1, 2)) / absolute,
            difference.max(axis=(1, 2)) / rms,
            correlation,
            nonfinite,
            repeated,
        )
    )


class LogisticBaseline:
    """L2 logistic regression with training-only standardization.

    Fixed regularization is a teaching choice, not tuned on test data.
    Objective averages Bernoulli negative log likelihood plus L2/2n.
    """

    def fit(self, features, labels):
        features, labels = np.asarray(features), np.asarray(labels)
        if features.ndim != 2 or labels.shape != (len(features),):
            raise ValueError("feature matrix and matching labels required")
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError("both binary classes required")
        self.center = features.mean(axis=0)
        self.scale = features.std(axis=0)
        self.scale[self.scale == 0] = 1
        x = np.column_stack(
            (np.ones(len(features)), (features - self.center) / self.scale)
        )

        def objective(weights):
            logits = x @ weights
            penalty = weights.copy()
            penalty[0] = 0
            loss = np.mean(np.logaddexp(0, logits) - labels * logits)
            loss += 0.5 * np.dot(penalty, penalty) / len(labels)
            gradient = x.T @ (expit(logits) - labels) / len(
                labels
            ) + penalty / len(labels)
            return loss, gradient

        result = minimize(
            objective,
            np.zeros(x.shape[1]),
            jac=True,
            method="L-BFGS-B",
            options={"maxiter": 1000, "gtol": 1e-8},
        )
        if not result.success:
            raise RuntimeError(
                f"baseline optimization failed: {result.message}"
            )
        self.weights = result.x
        return self

    def predict(self, features):
        x = np.column_stack(
            (
                np.ones(len(features)),
                (np.asarray(features) - self.center) / self.scale,
            )
        )
        return expit(x @ self.weights)

    def receipt(self):
        return dict(
            center=self.center.tolist(),
            scale=self.scale.tolist(),
            weights=self.weights.tolist(),
            l2_penalty=1,
            fitted_split="train",
        )


def development_baselines(raw_dir, prepared_dir, output):
    """Fit both baselines on training, freeze their validation thresholds."""
    import json
    from pathlib import Path

    from evaluation import confusion, paired_field_bootstrap, recall_threshold

    raw_dir, prepared_dir, output = map(Path, (raw_dir, prepared_dir, output))
    if output.exists():
        raise ValueError("baseline output must not exist")
    normalization = json.loads(
        (prepared_dir / "normalization.json").read_text()
    )
    selected = normalization["selected_raw_indices"]
    images = np.stack(
        [
            np.load(raw_dir / f"{name}.npy", mmap_mode="r")[selected]
            for name in ("search", "template", "difference")
        ],
        axis=1,
    )
    rows = [
        json.loads(line)
        for line in (prepared_dir / "metadata.jsonl").read_text().splitlines()
    ]
    features = measured_features(images, [r["snr"] for r in rows])
    split = np.load(prepared_dir / "split.npy")
    labels = np.load(prepared_dir / "labels.npy")
    fields = np.array([r["field"] for r in rows])
    output.mkdir(parents=True)
    receipts = {}
    for name, columns in (
        ("snr_only", [0]),
        ("measured_statistics", list(range(len(FEATURE_NAMES)))),
    ):
        model = LogisticBaseline().fit(
            features[split == 0][:, columns], labels[split == 0]
        )
        scores = model.predict(features[:, columns])
        validation = split == 1
        threshold = recall_threshold(labels[validation], scores[validation])
        interval = paired_field_bootstrap(
            labels[validation],
            scores[validation],
            scores[validation],
            fields[validation],
            threshold,
            threshold,
        )
        receipts[name] = dict(
            model=model.receipt(),
            threshold=threshold,
            feature_names=[FEATURE_NAMES[i] for i in columns],
            training_signature_audit={
                str(label): {
                    FEATURE_NAMES[i]: dict(
                        minimum=float(
                            features[
                                (split == 0) & (labels == label), i
                            ].min()
                        ),
                        maximum=float(
                            features[
                                (split == 0) & (labels == label), i
                            ].max()
                        ),
                        nonzero=int(
                            np.count_nonzero(
                                features[(split == 0) & (labels == label), i]
                            )
                        ),
                    )
                    for i in (-2, -1)
                }
                for label in (0, 1)
            },
            validation=confusion(
                labels[validation], scores[validation], threshold
            ),
            validation_fpr_interval=interval["a"],
        )
        np.savez(
            output / f"{name}.npz", scores=scores, labels=labels, split=split
        )
    (output / "baselines.json").write_text(
        json.dumps(receipts, indent=2) + "\n"
    )
    return receipts
