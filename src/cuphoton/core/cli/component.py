# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Component metadata and the fixed public command registry."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

_GROUP_RE = re.compile(r"^[a-z][a-z0-9]*$")


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    """Immutable command-line metadata for a cuPhoton component."""

    group: str
    import_name: str
    description: str
    group_help_description: str | None = None
    commands_module: str | None = None
    log_filename: str | None = None
    version_label: str | None = None
    show_equal_aliases: bool = False
    parser_style: Literal["optparse", "argparse"] = "optparse"
    group_help_style: Literal["subcommands", "commands"] = "subcommands"
    command_indent: int = 0
    supports_version_command: bool = True

    def __post_init__(self) -> None:
        if not _GROUP_RE.fullmatch(self.group):
            raise ValueError(
                "component group must start with a lowercase letter and "
                "contain only lowercase letters and digits"
            )
        import_suffix = self.import_name.removeprefix("cuphoton.")
        if (
            not self.import_name.startswith("cuphoton.")
            or not import_suffix
            or any(
                not part.isidentifier() for part in import_suffix.split(".")
            )
        ):
            raise ValueError(
                "component import_name must be within the cuphoton namespace"
            )
        if self.parser_style not in {"optparse", "argparse"}:
            raise ValueError(f"unknown parser style: {self.parser_style!r}")
        if self.group_help_style not in {"subcommands", "commands"}:
            raise ValueError(
                f"unknown group help style: {self.group_help_style!r}"
            )
        if self.command_indent < 0:
            raise ValueError("command_indent must be non-negative")

    @property
    def module_name(self) -> str:
        return self.commands_module or f"{self.import_name}.commands"

    @property
    def program_name(self) -> str:
        return f"cuphoton {self.group}"

    @property
    def app_dir(self) -> str:
        return self.group

    @property
    def log_env(self) -> str:
        group = self.group.upper().replace("-", "_")
        return f"CUPHOTON_{group}_LOG_FILE"

    @property
    def default_log_filename(self) -> str:
        return self.log_filename or f"{self.group}.log"


COMPONENTS = (
    ComponentSpec(
        "xdr",
        "cuphoton.xdr",
        "xDataReader: GPU-native FITS and HDF5 metadata loading.",
        parser_style="argparse",
        supports_version_command=False,
    ),
    ComponentSpec(
        "xfit",
        "cuphoton.xfit",
        "xFit: Batched nonlinear least-squares fitting for astronomical "
        "models.",
        version_label="",
    ),
    ComponentSpec(
        "xpois",
        "cuphoton.xpois",
        "Optimal image subtraction for astronomical images.",
        version_label="",
        show_equal_aliases=True,
    ),
    ComponentSpec(
        "xscan",
        "cuphoton.xscan",
        "Transient-candidate dataset, training, and review workflows.",
        version_label="",
    ),
    ComponentSpec(
        "xrep",
        "cuphoton.xrep",
        "xRep (xReproject): WCS-aware astronomical image reprojection "
        "and stacking.",
        version_label="",
    ),
    ComponentSpec(
        "xray",
        "cuphoton.xray",
        "GPU-accelerated X-ray detector analysis tools.",
        parser_style="argparse",
        group_help_style="commands",
        version_label="cuphoton xray",
    ),
)

COMPONENT_REGISTRY = {component.group: component for component in COMPONENTS}


def get_component(component: str | ComponentSpec) -> ComponentSpec:
    """Resolve public metadata or preserve an external specification."""

    if isinstance(component, ComponentSpec):
        return component
    try:
        return COMPONENT_REGISTRY[component]
    except KeyError as exc:
        raise KeyError(f"unknown cuPhoton component: {component!r}") from exc


__all__ = [
    "COMPONENT_REGISTRY",
    "COMPONENTS",
    "ComponentSpec",
    "get_component",
]
