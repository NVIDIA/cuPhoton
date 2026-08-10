# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from cuphoton import __version__

CORE_IMPORTS = (
    ("numpy", "array operations", "required"),
    ("h5py", "HDF5 input", "required"),
)

GPU_IMPORTS = (("cupy", "CUDA array kernels", "gpu"),)

OPTIONAL_IMPORTS = (("bokeh", "interactive review views", "optional"),)


@dataclass(frozen=True)
class ImportProbe:
    module: str
    purpose: str
    category: str
    available: bool


@dataclass(frozen=True)
class ExecutableProbe:
    name: str
    available: bool


@dataclass(frozen=True)
class DoctorReport:
    runtime: dict[str, Any]
    imports: tuple[ImportProbe, ...]
    executables: tuple[ExecutableProbe, ...]
    cuda_visibility: str

    def to_dict(self):
        return asdict(self)


def _module_available(name: str):
    return importlib.util.find_spec(name) is not None


def _probe_imports():
    probes = []
    for module, purpose, category in (
        CORE_IMPORTS + GPU_IMPORTS + OPTIONAL_IMPORTS
    ):
        probes.append(
            ImportProbe(
                module,
                purpose,
                category,
                _module_available(module),
            )
        )
    return tuple(probes)


def _runtime_metadata() -> dict[str, Any]:
    runtime: dict[str, Any] = {
        "python_version": platform.python_version(),
        "cuphoton_version": __version__,
        "numpy_version": np.__version__,
        "cupy": {"available": False, "devices": []},
    }
    try:
        import cupy as cp

        device_count = int(cp.cuda.runtime.getDeviceCount())
        devices = []
        for index in range(device_count):
            properties = cp.cuda.runtime.getDeviceProperties(index)
            name = properties.get("name", "unknown")
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            devices.append(
                {
                    "index": index,
                    "name": str(name),
                    "compute_capability": (
                        f"{properties.get('major')}.{properties.get('minor')}"
                    ),
                    "total_memory_bytes": int(
                        properties.get("totalGlobalMem", 0)
                    ),
                }
            )
        runtime["cupy"] = {
            "available": True,
            "version": str(cp.__version__),
            "cuda_driver_version": int(cp.cuda.runtime.driverGetVersion()),
            "cuda_runtime_version": int(cp.cuda.runtime.runtimeGetVersion()),
            "devices": devices,
        }
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover - runtime-specific
        runtime["cupy"] = {
            "available": True,
            "devices": [],
            "error": f"{type(exc).__name__}: {exc}",
        }
    return runtime


def _probe_executables():
    return tuple(
        ExecutableProbe(name=name, available=shutil.which(name) is not None)
        for name in ("nvidia-smi", "sbatch")
    )


def collect_doctor_report(environ: dict[str, str] | None = None):
    env = os.environ if environ is None else environ
    visible = env.get("CUDA_VISIBLE_DEVICES")
    if visible is None:
        visibility = "unset"
    elif visible.strip():
        visibility = "set"
    else:
        visibility = "empty"
    return DoctorReport(
        runtime=_runtime_metadata(),
        imports=_probe_imports(),
        executables=_probe_executables(),
        cuda_visibility=visibility,
    )


def format_doctor_text(report: DoctorReport):
    lines = ["xray doctor"]
    lines.append("")
    lines.append("Runtime:")
    lines.append(f"  Python: {report.runtime['python_version']}")
    lines.append(f"  cuPhoton: {report.runtime['cuphoton_version']}")
    lines.append(f"  NumPy: {report.runtime['numpy_version']}")
    cupy = report.runtime["cupy"]
    if cupy.get("available"):
        lines.append(f"  CuPy: {cupy.get('version', 'unusable')}")
        lines.append(
            "  CUDA driver/runtime: "
            f"{cupy.get('cuda_driver_version', '-')} / "
            f"{cupy.get('cuda_runtime_version', '-')}"
        )
        for device in cupy.get("devices", []):
            memory_gib = device["total_memory_bytes"] / 1024**3
            lines.append(
                f"  GPU {device['index']}: {device['name']} "
                f"(cc {device['compute_capability']}, {memory_gib:.1f} GiB)"
            )
        if cupy.get("error"):
            lines.append(f"  CuPy error: {cupy['error']}")
    else:
        lines.append("  CuPy: not installed")
    lines.append("")
    lines.append("Capabilities:")
    for probe in report.imports:
        state = "ok" if probe.available else "missing"
        lines.append(
            f"  {state:7} {probe.module:14} "
            f"[{probe.category}] {probe.purpose}"
        )
    lines.append("")
    lines.append("Executables:")
    for probe in report.executables:
        state = "ok" if probe.available else "missing"
        lines.append(f"  {state:7} {probe.name}")
    lines.append("")
    lines.append(f"CUDA visibility: {report.cuda_visibility}")
    return "\n".join(lines)


def format_doctor_json(report: DoctorReport):
    return json.dumps(report.to_dict(), indent=2, sort_keys=True)
