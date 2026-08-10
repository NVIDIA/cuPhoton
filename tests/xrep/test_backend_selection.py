# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from types import SimpleNamespace

from cuphoton.xrep import backends


def test_auto_backend_prefers_cupy_then_cuda_torch(monkeypatch) -> None:
    available = {"cupy": False, "torch": True, "cpu": True}
    seen: list[str] = []

    def is_available(name: str) -> bool:
        seen.append(name)
        return available[name]

    monkeypatch.setattr(backends, "_backend_available_for_auto", is_available)

    assert backends.default_backend() == "torch"
    assert seen == ["cupy", "torch"]

    available["cupy"] = True
    seen.clear()
    assert backends.default_backend() == "cupy"
    assert seen == ["cupy"]


def test_auto_backend_falls_back_to_cpu_without_gpu(monkeypatch) -> None:
    monkeypatch.setattr(
        backends,
        "_backend_available_for_auto",
        lambda name: name == "cpu",
    )

    assert backends.default_backend() == "cpu"


def test_torch_requires_cuda_only_for_auto_selection(monkeypatch) -> None:
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False)
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    assert backends._backend_available_for_auto("torch") is False
    assert backends.get_backend("torch").is_available() is True


def test_explicit_auto_resolves_to_default_backend(monkeypatch) -> None:
    monkeypatch.setattr(backends, "default_backend", lambda: "cpu")

    assert backends.resolve_backend("auto") == "cpu"
    assert backends.get_backend("auto").name == "cpu"
