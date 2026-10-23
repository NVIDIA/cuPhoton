# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Robust losses and iteratively reweighted least-squares weights."""

from __future__ import annotations

from typing import Any

ROBUST_LOSSES = frozenset(
    {
        "huber",
        "cauchy",
        "bisquare",
        "damped_rl",
    }
)


def rho_values(
    torch: Any,
    residual: Any,
    *,
    loss: str,
    delta: Any,
    variance: Any,
) -> Any:
    """Evaluate a configured robust loss elementwise."""

    _validate_loss(loss)
    if loss == "huber":
        return torch.where(
            torch.abs(residual) > delta,
            delta * (torch.abs(residual) - 0.5 * delta),
            0.5 * residual.square(),
        )
    if loss == "cauchy":
        return 0.5 * delta.square() * torch.log1p((residual / delta).square())
    if loss == "bisquare":
        scaled = residual
        return torch.where(
            torch.abs(scaled) > 1.0,
            torch.ones_like(scaled),
            1.0 - (1.0 - scaled.square()).pow(3),
        )
    transformed = _damped_rl_g(torch, residual)
    return 0.5 * transformed.square()


def irls_weights(
    torch: Any,
    residual: Any,
    *,
    loss: str,
    delta: Any,
    variance: Any,
) -> Any:
    """Evaluate robust IRLS weights elementwise."""

    _validate_loss(loss)
    if loss == "huber":
        safe = torch.where(
            torch.abs(residual) > 1e-12,
            residual,
            torch.ones_like(residual),
        )
        return torch.where(
            torch.abs(residual) > delta,
            delta * torch.sign(residual) / safe,
            torch.ones_like(residual),
        )
    if loss == "cauchy":
        return 1.0 / (1.0 + (residual / delta).square())
    if loss == "bisquare":
        return torch.where(
            torch.abs(residual) > 1.0,
            torch.zeros_like(residual),
            (1.0 - residual.square()).square(),
        )
    numerator = _damped_rl_g(torch, residual) * _damped_rl_gprime(
        torch, residual
    )
    return torch.where(
        torch.abs(residual) > 1e-12,
        numerator / residual,
        torch.ones_like(residual),
    )


def _damped_rl_g(torch: Any, residual: Any, *, order: int = 10) -> Any:
    magnitude = torch.abs(residual)
    coefficient = (order - 1.0) / (order + 1.0)
    transformed = torch.where(
        magnitude > 1.0,
        magnitude,
        magnitude.pow(order)
        - coefficient * magnitude.pow(order + 1)
        + coefficient,
    )
    return transformed


def _damped_rl_gprime(torch: Any, residual: Any, *, order: int = 10) -> Any:
    magnitude = torch.abs(residual)
    derivative = torch.where(
        magnitude > 1.0,
        torch.ones_like(magnitude),
        order * magnitude.pow(order - 1) - (order - 1) * magnitude.pow(order),
    )
    return derivative * torch.sign(residual)


def _validate_loss(loss: str) -> None:
    if loss not in ROBUST_LOSSES:
        choices = ", ".join(sorted(ROBUST_LOSSES))
        raise ValueError(
            f"Unsupported robust loss {loss!r}; choose {choices}."
        )
