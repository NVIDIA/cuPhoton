# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Backend registry."""


def get_backend(name: str):
    if name in {"auto", "torch"}:
        from .torch_backend import TorchBackend

        return TorchBackend()
    raise ValueError(
        f"Unsupported backend {name!r}; choose 'auto' or 'torch'."
    )
