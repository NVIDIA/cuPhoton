# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Text output helpers for XRay commands."""

from __future__ import annotations


def print_file_probe(label, probe):
    print(f"{label}_path={probe.path}")
    print(f"{label}_size_bytes={probe.size_bytes}")
    print(f"{label}_schema={probe.schema}")
    print(f"{label}_ipm_pairs={','.join(probe.ipm_pairs) or '-'}")
    print(f"{label}_keys={','.join(probe.keys)}")
    for dataset in probe.datasets:
        shape = "x".join(str(dim) for dim in dataset.shape)
        chunks = (
            "-"
            if dataset.chunks is None
            else "x".join(str(dim) for dim in dataset.chunks)
        )
        print(
            f"{label}_dataset={dataset.name} "
            f"shape={shape} dtype={dataset.dtype} chunks={chunks}"
        )


def print_validation_sweep(sweep):
    print(f"estimator={sweep.backend}")
    if sweep.distortion is not None:
        print(f"distortion={sweep.distortion[0]}:{sweep.distortion[1]:g}")
    for level in sweep.levels:
        print(
            f"snr_db={level.snr_db:g} sigma={level.noise_sigma:.4g} "
            f"trials={level.trials_successful}/{level.trials_attempted} "
            f"estimator_errors={level.estimator_errors} "
            f"any_mode_lost_rate={level.any_mode_lost_rate:.3f} "
            f"residual_ratio={level.residual_ratio:.2f}"
        )
        for m in level.modes:
            w = m.angular_frequency
            d = m.decay
            print(
                f"  mode w={w.truth:g} decay={d.truth:g}: "
                f"freq std/crlb={w.std:.3g}/{w.crlb_std:.3g} "
                f"({w.std_over_crlb_std:.2f}x) bias={w.bias:+.3g}; "
                f"decay std/crlb={d.std:.3g}/{d.crlb_std:.3g} "
                f"({d.std_over_crlb_std:.2f}x) bias={d.bias:+.3g}; "
                f"recovered={m.recovered_trials} "
                f"loss_rate={m.loss_rate:.3f}"
            )


def print_model_order_sweep_payload(payload):
    print(f"source={payload['source']['kind']}")
    print(f"samples={payload['samples']}")
    print(f"roots_backend={payload['roots_backend']}")
    print(f"best_components={payload['best_components']}")
    print(f"best_selected_model_order={payload['best_selected_model_order']}")
    print(f"best_rms_residual={payload['best_rms_residual']:.6g}")
    print(
        "best_reconstruction_rms_error="
        f"{payload['best_reconstruction_rms_error']:.6g}"
    )
    for entry in payload["entries"]:
        print(
            f"components={entry['components']} "
            f"selected_model_order={entry['selected_model_order']} "
            f"rms_residual={entry['rms_residual']:.6g} "
            "reconstruction_rms_error="
            f"{entry['reconstruction_rms_error']:.6g} "
            f"chi2={entry['chi2']:.6g} "
            f"selected_roots={entry['selected_root_count']} "
            f"decaying_roots={entry['decaying_root_count']}"
        )


def print_model_order_sweep_batch(payload):
    print(f"source={payload['source']['kind']}")
    print(f"trace_count={payload['trace_count']}")
    print(f"samples={payload['samples']}")
    print(f"roots_backend={payload['roots_backend']}")
    component_counts = ",".join(
        str(item) for item in payload["component_counts"]
    )
    print(f"component_counts={component_counts}")
    print(
        "best_components_unique="
        + ",".join(str(item) for item in payload["best_components_unique"])
    )
    print(
        "best_selected_model_orders_unique="
        + ",".join(
            str(item) for item in payload["best_selected_model_orders_unique"]
        )
    )
    for trace_payload in payload["traces"]:
        print(f"trace_index={trace_payload['trace_index']}")
        print(f"  best_components={trace_payload['best_components']}")
        print(
            "  best_selected_model_order="
            f"{trace_payload['best_selected_model_order']}"
        )
        print(
            "  best_reconstruction_rms_error="
            f"{trace_payload['best_reconstruction_rms_error']:.6g}"
        )


def print_subspace_benchmark_batch(payload):
    print(f"source={payload['source']['kind']}")
    print(f"trace_count={payload['trace_count']}")
    print(f"samples={payload['samples']}")
    print(f"model_order={payload['model_order']}")
    print(f"components={payload['components']}")
    print(
        "baseline_rms_residual_range="
        f"{payload['baseline_rms_residual_min']:.6g}:"
        f"{payload['baseline_rms_residual_max']:.6g}"
    )
    for summary in payload["method_summary"]:
        print(f"method={summary['method']}")
        print(f"  svd_backend={summary['svd_backend']}")
        print(f"  trace_count={summary['trace_count']}")
        print(
            "  rms_residual_range="
            f"{summary['rms_residual_min']:.6g}:"
            f"{summary['rms_residual_max']:.6g}"
        )
        print(
            "  max_abs_reconstruction_diff_max="
            f"{summary['max_abs_reconstruction_diff_max']:.6g}"
        )
