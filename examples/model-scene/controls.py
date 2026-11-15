# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Workflow-level controls for units, weights and CPU/CUDA agreement."""

import numpy as np
from experiment import forward_prediction, reconstruct
from scene import generate


def numerical_controls(root, *, device="cpu"):
    """Exercise actual workflow preprocessing and solvers on paired inputs."""
    root.mkdir(parents=True, exist_ok=False)
    arrays, _ = generate(1901, development=True, matched=True)

    def fit(name, data, algorithm="l2", delta=2):
        return reconstruct(
            root,
            data,
            name=name,
            device=device,
            algorithm=algorithm,
            iterations=30,
            delta=delta,
        )[0]

    baseline = fit("rates", arrays)
    times = np.linspace(10, 40, 12)
    counts = {key: value.copy() for key, value in arrays.items()}
    counts["images"] *= times
    counts["variances"] *= times**2
    counts["exp_times"] = times
    normalized = fit("counts", counts)
    np.testing.assert_allclose(normalized, baseline, rtol=1e-10, atol=1e-10)
    scaled_variance = {key: value.copy() for key, value in arrays.items()}
    scaled_variance["variances"] *= 4
    weighted = fit("variance-four", scaled_variance)
    np.testing.assert_allclose(
        weighted[7:-7, 7:-7], baseline[7:-7, 7:-7], rtol=1e-10, atol=1e-10
    )
    np.testing.assert_allclose(
        forward_prediction(weighted, arrays["psfs"]),
        forward_prediction(baseline, arrays["psfs"]),
        rtol=1e-8,
        atol=1e-8,
    )
    robust_scaled = fit("robust-variance-four", scaled_variance, "rho")
    robust_delta = fit("robust-delta-four", arrays, "rho", delta=4)
    np.testing.assert_allclose(
        robust_scaled[7:-7, 7:-7],
        robust_delta[7:-7, 7:-7],
        rtol=1e-10,
        atol=1e-10,
    )
    return {
        "l2_variance_scale_roi_max_abs": float(
            np.max(np.abs(weighted[7:-7, 7:-7] - baseline[7:-7, 7:-7]))
        ),
        "l2_variance_scale_prediction_max_abs": float(
            np.max(
                np.abs(
                    forward_prediction(weighted, arrays["psfs"])
                    - forward_prediction(baseline, arrays["psfs"])
                )
            )
        ),
        "huber_variance_delta_roi_max_abs": float(
            np.max(
                np.abs(robust_scaled[7:-7, 7:-7] - robust_delta[7:-7, 7:-7])
            )
        ),
        "counts_rates_max_abs": float(np.max(np.abs(normalized - baseline))),
        "l2_variance_scale_full_halo_max_abs": float(
            np.max(np.abs(weighted - baseline))
        ),
        "huber_variance_delta_full_halo_max_abs": float(
            np.max(np.abs(robust_scaled - robust_delta))
        ),
    }


def asymmetric_operator_control(root):
    """Check torch preprocessing against asymmetric truth."""
    import torch
    from scene import write_bundle
    from scipy.signal import convolve2d

    from cuphoton.ximage.backends.torch_gpu_preprocess import (
        prepare_torch_reconstruction_inputs,
    )
    from cuphoton.ximage.config import ReconstructionConfig
    from cuphoton.ximage.data import load_ximage_bundle

    arrays, truth = generate(1902, development=True, matched=True)
    psfs = arrays["psfs"].copy()
    psfs[8, 10, 0, :] += 0.1
    psfs /= psfs.sum(axis=(0, 1), keepdims=True)
    arrays["psfs"] = psfs
    arrays["images"] = np.stack(
        [
            convolve2d(truth["truth"], psfs[:, :, 0, i], mode="same")
            for i in range(12)
        ],
        axis=-1,
    )[None]
    write_bundle(root, arrays)
    config = ReconstructionConfig(
        algorithm="l2", device="cpu", roi=dict(height=48, width=48)
    )
    with load_ximage_bundle(root) as bundle:
        prepared = prepare_torch_reconstruction_inputs(
            bundle,
            config,
            device=torch.device("cpu"),
            dtype=torch.float64,
            psfs=psfs,
        )
    latent = torch.tensor(truth["truth"][5:67, 5:67])[None, None]
    predicted = torch.nn.functional.conv2d(latent, prepared.f)
    np.testing.assert_allclose(
        predicted.numpy(), prepared.y.numpy(), atol=1e-12, rtol=1e-12
    )
    wrong = torch.nn.functional.conv2d(latent, prepared.f.flip((-2, -1)))
    transposed = torch.nn.functional.conv2d(
        latent, prepared.f.transpose(-2, -1)
    )
    permuted = torch.nn.functional.conv2d(latent, prepared.f.flip((0,)))
    errors = {
        "correct_max_abs": float((predicted - prepared.y).abs().max()),
        "flipped_max_abs": float((wrong - prepared.y).abs().max()),
        "transposed_max_abs": float((transposed - prepared.y).abs().max()),
        "frame_permutation_max_abs": float(
            (permuted - prepared.y).abs().max()
        ),
    }
    assert min(errors[key] for key in errors if key != "correct_max_abs") > 1
    return errors
