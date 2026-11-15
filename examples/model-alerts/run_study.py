# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the development pilot; final test seeds are never accessed here."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path

import numpy as np
from baselines import development_baselines
from generator import Design, write_dataset
from study import prepare_training, train_development


def pilot(output, *, smoke=False, device="cuda:0", multiplier=1):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("pilot output must not exist")
    output.mkdir(parents=True)
    train_count, val_count = (6, 3) if smoke else (180, 60)
    train_count *= multiplier
    val_count *= multiplier
    fields = {i: "train" for i in range(train_count)}
    fields.update({10000 + i: "val" for i in range(val_count)})
    raw, prepared = output / "raw", output / "prepared"
    design = Design()
    write_dataset(raw, fields, design=design, backend="cpu")
    raw_rows = [
        json.loads(line)
        for line in (raw / "metadata.jsonl").read_text().splitlines()
    ]
    difference = np.load(raw / "difference.npy", mmap_mode="r")
    mask = np.array(
        [
            r["family"] == "misregistration"
            and not r["label"]
            and not r["ambiguous"]
            for r in raw_rows
        ]
    )
    center = design.stamp_size // 2
    cut = np.s_[center - 4 : center + 5, center - 4 : center + 5]
    selected = difference[mask][:, cut[0], cut[1]]
    variance = np.load(raw / "variance.npy", mmap_mode="r")[mask][
        :, cut[0], cut[1]
    ]
    standardized = selected / np.sqrt(variance)
    bipolar = (standardized.max(axis=(1, 2)) >= 5) & (
        standardized.min(axis=(1, 2)) <= -5
    )
    registration_diagnostic = dict(
        candidates=int(mask.sum()),
        bipolar_candidates=int(bipolar.sum()),
        criterion=(
            "central 9x9 has both signs at >=5 marginal pixel sigma; "
            "descriptive, not calibrated detection significance"
        ),
    )
    if not smoke and not bipolar.any():
        raise RuntimeError(
            "development contains no bipolar registration residuals"
        )
    prepare_training(raw, prepared)
    baselines = development_baselines(raw, prepared, output / "baselines")
    outcomes = []
    for mode in ("pair", "triplet"):
        for seed in (11,) if smoke else (11, 23, 47):
            print(f"Training {mode}, seed {seed}", flush=True)
            outcomes.append(
                train_development(
                    prepared,
                    output / f"{mode}-{seed}",
                    mode=mode,
                    seed=seed,
                    epochs=1 if smoke else 12,
                    device=device,
                )
            )
    half_widths = [
        (r["validation_fpr_interval"][1] - r["validation_fpr_interval"][0])
        / 2
        for r in outcomes
    ]
    receipt = dict(
        profile="quick-smoke" if smoke else "development",
        scientific_performance_qualified=False,
        train_fields=train_count,
        validation_fields=val_count,
        epochs_per_field=design.epochs,
        outcomes=outcomes,
        baselines=baselines,
        precision_requires_expansion=max(half_widths) > 0.05,
        maximum_validation_half_width=max(half_widths),
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(__file__).parent.glob("*.py")
        },
        registration_diagnostic=registration_diagnostic,
        final_test_generated=False,
    )
    (output / "pilot.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=["quick-smoke", "development"],
        default="quick-smoke",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-lock", type=Path)
    args = parser.parse_args()
    if args.device.startswith("cuda"):
        if args.gpu_lock is None:
            parser.error("CUDA runs require an explicit shared --gpu-lock")
        with args.gpu_lock.open("a") as lock:
            print("Waiting for shared GPU lock", flush=True)
            fcntl.flock(lock, fcntl.LOCK_EX)
            print("Acquired shared GPU lock", flush=True)
            pilot(
                args.output,
                smoke=args.profile == "quick-smoke",
                device=args.device,
            )
    else:
        pilot(
            args.output,
            smoke=args.profile == "quick-smoke",
            device=args.device,
        )


if __name__ == "__main__":
    main()
