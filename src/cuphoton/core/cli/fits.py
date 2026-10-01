# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional FITS reader overrides for manifest-driven commands."""

from .invariants import SetInvariant


class FitsReaderOptions:
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
