# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from cuphoton.ximage.robust import ROBUST_LOSSES, irls_weights, rho_values

torch = pytest.importorskip("torch")


@pytest.mark.parametrize(
    ("loss", "expected_rho", "expected_weights"),
    [
        ("huber", [0.0, 0.125, 1.5], [1.0, 1.0, 0.5]),
        (
            "cauchy",
            [0.0, 0.11157177565710488, 0.8047189562170501],
            [1.0, 0.8, 0.2],
        ),
        ("bisquare", [0.0, 0.578125, 1.0], [1.0, 0.5625, 0.0]),
        (
            "damped_rl",
            [0.33471074380165294, 0.33518305002165244, 2.0],
            [0.0, 0.01759052276611328, 1.0],
        ),
    ],
)
def test_robust_formulas_match_reference_values(
    loss: str,
    expected_rho: list[float],
    expected_weights: list[float],
) -> None:
    residual = torch.tensor([0.0, 0.5, 2.0], dtype=torch.float64)
    variance = torch.full_like(residual, 4.0)
    delta = torch.ones_like(residual)

    rho = rho_values(
        torch,
        residual,
        loss=loss,
        delta=delta,
        variance=variance,
    )
    weights = irls_weights(
        torch,
        residual,
        loss=loss,
        delta=delta,
        variance=variance,
    )

    torch.testing.assert_close(
        rho,
        torch.tensor(expected_rho, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        weights,
        torch.tensor(expected_weights, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("loss", sorted(ROBUST_LOSSES))
def test_robust_loss_and_weights_are_finite(loss: str) -> None:
    residual = torch.tensor([-1e6, -1.0, 0.0, 1.0, 1e6], dtype=torch.float64)
    variance = torch.ones_like(residual)
    delta = torch.ones_like(residual)

    rho = rho_values(
        torch,
        residual,
        loss=loss,
        delta=delta,
        variance=variance,
    )
    weights = irls_weights(
        torch,
        residual,
        loss=loss,
        delta=delta,
        variance=variance,
    )

    assert torch.isfinite(rho).all()
    assert torch.isfinite(weights).all()
    assert torch.all(rho >= 0.0)
    assert torch.all(weights >= 0.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_damped_rl_weights_match_loss_derivative_with_finite_gradients(
    dtype,
) -> None:
    residual = torch.tensor(
        [-1e6, -1.0, -0.5, -1e-3, 0.0, 1e-3, 0.5, 1.0, 1e6],
        dtype=dtype,
        requires_grad=True,
    )
    kwargs = {
        "loss": "damped_rl",
        "delta": torch.ones_like(residual),
        "variance": torch.ones_like(residual),
    }
    rho = rho_values(torch, residual, **kwargs)
    weights = irls_weights(torch, residual, **kwargs)
    loss_derivative = torch.autograd.grad(rho.sum(), residual)[0]
    weight_derivative = torch.autograd.grad(weights.sum(), residual)[0]

    assert torch.isfinite(loss_derivative).all()
    assert torch.isfinite(weight_derivative).all()
    torch.testing.assert_close(loss_derivative, weights * residual)
    # The damping weight tends to zero near an exactly fitted sample.
    assert weights[4] == 0.0
    assert torch.all(weights[3:6] < 1e-20)


@pytest.mark.parametrize(
    "loss",
    ["bisquare_scale", "gauss_mixture", "gauss_uniform_mixture", "unknown"],
)
def test_robust_helpers_reject_unsupported_losses(loss: str) -> None:
    residual = torch.tensor([0.5], dtype=torch.float64)
    for function in (rho_values, irls_weights):
        with pytest.raises(ValueError, match="Unsupported robust loss"):
            function(
                torch,
                residual,
                loss=loss,
                delta=torch.ones_like(residual),
                variance=torch.ones_like(residual),
            )
