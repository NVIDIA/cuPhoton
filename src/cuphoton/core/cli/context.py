# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Side-effect-free XDG application path resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .component import ComponentSpec, get_component


@dataclass(frozen=True, slots=True)
class ApplicationContext:
    """Resolved filesystem context for one command component."""

    component: ComponentSpec
    config_home: Path
    state_home: Path
    data_home: Path
    log_file_override: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "component", get_component(self.component))
        for name in ("config_home", "state_home", "data_home"):
            path = Path(getattr(self, name)).expanduser().resolve()
            object.__setattr__(self, name, path)
        if self.log_file_override is not None:
            override = Path(self.log_file_override).expanduser().resolve()
            object.__setattr__(self, "log_file_override", override)

    @property
    def config_dir(self) -> Path:
        return self.config_home / "cuphoton" / self.component.app_dir

    @property
    def state_dir(self) -> Path:
        return self.state_home / "cuphoton" / self.component.app_dir

    @property
    def data_dir(self) -> Path:
        return self.data_home / "cuphoton" / self.component.app_dir

    @property
    def workspace_dir(self) -> Path:
        return self.state_dir

    @property
    def runs_dir(self) -> Path:
        return self.state_dir / "runs"

    @property
    def logs_dir(self) -> Path:
        return self.state_dir / "logs"

    @property
    def log_file(self) -> Path:
        return (
            self.log_file_override
            if self.log_file_override is not None
            else self.logs_dir / self.component.default_log_filename
        )

    @classmethod
    def for_component(
        cls,
        component: str | ComponentSpec,
        environ: Mapping[str, str] | None = None,
    ) -> ApplicationContext:
        """Resolve component paths without creating filesystem entries."""

        spec = get_component(component)
        env = os.environ if environ is None else environ
        home = Path(env.get("HOME") or Path.home()).expanduser()
        config_home = Path(
            env.get("XDG_CONFIG_HOME") or home / ".config"
        ).expanduser()
        state_home = Path(
            env.get("XDG_STATE_HOME") or home / ".local" / "state"
        ).expanduser()
        data_home = Path(
            env.get("XDG_DATA_HOME") or home / ".local" / "share"
        ).expanduser()

        override = env.get(spec.log_env)
        return cls(
            component=spec,
            config_home=config_home,
            state_home=state_home,
            data_home=data_home,
            log_file_override=Path(override) if override else None,
        )


__all__ = ["ApplicationContext"]
