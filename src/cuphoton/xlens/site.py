# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Load an installed xLens application from a trusted local manifest."""

from __future__ import annotations

import importlib
import re
import tomllib
from pathlib import Path


def normalize_prefix(prefix: str) -> str:
    """Validate a URL mount and return its canonical trailing-slash form."""
    if not isinstance(prefix, str) or not re.fullmatch(
        r"/(?:[A-Za-z0-9._~-]+/)*[A-Za-z0-9._~-]*", prefix
    ):
        raise ValueError("site prefix must be an absolute URL path")
    if any(part in {".", ".."} for part in prefix.split("/")):
        raise ValueError("site prefix cannot contain dot path segments")
    return prefix.rstrip("/") + "/"


def serve_site(
    path: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
    prefix: str | None = None,
) -> None:
    """Serve an installed application named by ``xlens-site.toml``.

    The manifest is trusted local configuration. Its ``site.application``
    names an installed ``module:callable``; the callable owns its server
    lifecycle. No source directory is added to Python's import path.
    """
    manifest = Path(path).resolve()
    if manifest.is_dir():
        manifest /= "xlens-site.toml"
    with manifest.open("rb") as stream:
        config = tomllib.load(stream)
    site = config.get("site")
    if not isinstance(site, dict):
        raise ValueError("site manifest requires a [site] table")
    application = site.get("application")
    if not isinstance(application, str) or application.count(":") != 1:
        raise ValueError(
            "site.application must name an installed module:callable"
        )
    module_name, attribute = application.split(":")
    identifiers = (*module_name.split("."), attribute)
    if not all(part.isidentifier() for part in identifiers):
        raise ValueError(
            "site.application must name an installed module:callable"
        )
    mount = normalize_prefix(
        prefix if prefix is not None else site.get("prefix", "/")
    )
    try:
        callback = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as exc:
        raise ValueError(
            f"cannot load site application {application!r}: {exc}; "
            "check that its package and dependencies are installed"
        ) from exc
    if not callable(callback):
        raise ValueError("site.application must refer to a callable")
    callback(
        root=manifest.parent,
        config=config,
        host=host,
        port=port,
        prefix=mount,
    )
