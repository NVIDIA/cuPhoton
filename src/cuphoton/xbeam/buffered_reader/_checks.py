# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Shared validation for buffered-reader coordinates."""

from numbers import Integral

import numpy as np

from ..missing_timestamp_recovery import MissingTimestampRecoveryError


class BufferedWindowError(MissingTimestampRecoveryError):
    """The physical stream cannot satisfy the buffered-reader contract."""


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int and (
        not isinstance(value, Integral) or isinstance(value, bool)
    ):
        raise TypeError(f"{name} must be an integer")
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return int(value)


def _readonly(array: np.ndarray) -> np.ndarray:
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(
        array.shape
    )
