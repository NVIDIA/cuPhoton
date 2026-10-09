# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only validation of optional xDR runtime choices."""

from collections.abc import Mapping

XDR_OPTION_CHOICES = {
    "postprocess": frozenset({"auto", "fused", "separate"}),
    "gzip_decoder": frozenset({"auto", "gzip", "deflate"}),
    "decompression_backend": frozenset({"auto", "cuda"}),
}


def normalize_xdr_options(
    options: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy explicit choices, leaving omitted choices to xDR defaults."""
    if options is None:
        return {}
    if not isinstance(options, Mapping):
        raise ValueError("xdr_options must be a mapping")
    result = {}
    for name, value in options.items():
        if name not in XDR_OPTION_CHOICES:
            raise ValueError(f"unsupported xDR option: {name!r}")
        if (
            not isinstance(value, str)
            or value not in XDR_OPTION_CHOICES[name]
        ):
            choices = ", ".join(sorted(XDR_OPTION_CHOICES[name]))
            raise ValueError(f"xDR {name} must be one of: {choices}")
        result[name] = value
    return result


def merge_xdr_options(
    base: Mapping[str, str] | None,
    overrides: Mapping[str, str] | None,
) -> dict[str, str]:
    """Override only supplied keys and preserve other manifest choices."""
    return {**normalize_xdr_options(base), **normalize_xdr_options(overrides)}
