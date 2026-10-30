# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Public exception hierarchy for xbeam."""

from __future__ import annotations


class XBeamError(Exception):
    """Base class for errors intended to cross the package boundary."""


class JobLoadError(XBeamError):
    """A normalized job document could not be decoded."""


class ConfigurationError(XBeamError, ValueError):
    """A job contains an invalid static configuration.

    Parameters are kept separately so callers can report structured errors
    without parsing the human-readable message.
    """

    def __init__(self, entity: str, field: str, reason: str) -> None:
        self.entity = entity
        self.field = field
        self.reason = reason
        super().__init__(f"{entity}.{field}: {reason}")


class UnsupportedFeatureError(ConfigurationError):
    """A well-formed job requests behavior not supported by this release."""


class BackendUnavailableError(XBeamError, RuntimeError):
    """The requested numerical backend cannot be used in this process."""


class UnsupportedProfileError(UnsupportedFeatureError):
    """A requested compatibility profile is not frozen or implemented."""

    def __init__(self, reason: str) -> None:
        super().__init__("correlate", "profile", reason)
