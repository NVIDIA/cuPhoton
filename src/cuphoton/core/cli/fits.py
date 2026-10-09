# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional FITS reader overrides shared by FITS-consuming commands."""

from cuphoton.core.fits_options import (
    XDR_OPTION_CHOICES,
    normalize_xdr_options,
)

from .invariants import SetInvariant


class XdrOptionsMixin:
    """Optional xDR choices shared by FITS-consuming commands."""

    xdr_postprocess = None

    class XdrPostprocessArg(SetInvariant):
        _arg = "--xdr-postprocess"
        _help = (
            "xDR postprocessing: auto, fused, or separate. Omitted preserves "
            "input choices or uses xDR defaults; applies when the FITS "
            "reader uses xDR."
        )
        _set = XDR_OPTION_CHOICES["postprocess"]
        _default = None

    xdr_gzip_decoder = None

    class XdrGzipDecoderArg(SetInvariant):
        _arg = "--xdr-gzip-decoder"
        _help = (
            "xDR gzip decoder: auto, gzip, or deflate. Omitted preserves "
            "manifest choices; applies when the FITS reader uses xDR."
        )
        _set = XDR_OPTION_CHOICES["gzip_decoder"]
        _default = None


def xdr_options_from_cli(command) -> dict[str, str]:
    """Collect explicit flags without replacing omitted manifest choices."""
    return normalize_xdr_options(
        {
            name: value
            for name in XDR_OPTION_CHOICES
            if (value := getattr(command, f"xdr_{name}", None)) is not None
        }
    )


def xdr_option_cli_args(options) -> list[str]:
    """Serialize explicit options for a component's benchmark subprocess."""
    return [
        argument
        for name, value in normalize_xdr_options(options).items()
        for argument in (f"--xdr-{name.replace('_', '-')}", value)
    ]


class FitsReaderOptions(XdrOptionsMixin):
    """Preserve manifest policies unless a reader is explicitly selected."""

    fits_reader = None

    class FitsReaderArg(SetInvariant):
        _arg = "--fits-reader"
        _help = (
            "FITS reader: auto, astropy, or xdr. Overrides manifest reader "
            "policies for this run; omitted preserves them."
        )
        _set = {"auto", "astropy", "xdr"}
        _default = None
