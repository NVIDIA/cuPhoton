# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Optional GPU sorting and bounds reduction for large unordered mappings.

Only compact frontier bounds return to the host. Uploads retain the original
physical-order mappings; this module never sorts voltage payloads.
"""

from __future__ import annotations

import numpy as np

from .profiling import nvtx_range

_GPU_MIN_FRAMES = 262_144
_GPU_BYTES_PER_FRAME = 96  # Inputs, permutation, gathers, and sort workspace.


def gpu_frontier_bounds(physical, destinations, stops, xp, mode, diagnostics):
    """Return compact bounds, or None to retain the CPU planning path."""
    count = int(destinations.size)
    reason = None
    if mode == "cpu":
        reason = "disabled"
    elif xp is None or xp.__name__ != "cupy":
        reason = "requires-cupy"
    elif mode == "auto" and count < _GPU_MIN_FRAMES:
        reason = "small-mapping"
    if reason is None:
        free, total = xp.cuda.runtime.memGetInfo()
        available = min(
            int(total),
            int(free) + int(xp.get_default_memory_pool().free_bytes()),
        )
        if count * _GPU_BYTES_PER_FRAME > available // 10:
            reason = "device-budget"
    if reason is not None:
        diagnostics.setdefault("gpu_planner_fallbacks", {})[reason] = (
            diagnostics.get("gpu_planner_fallbacks", {}).get(reason, 0) + 1
        )
        return None
    diagnostics["gpu_planner_peak_estimated_bytes"] = max(
        diagnostics.get("gpu_planner_peak_estimated_bytes", 0),
        count * _GPU_BYTES_PER_FRAME,
    )
    try:
        with nvtx_range("xbeam:vdif-frontier-gpu-plan"):
            device_destinations = xp.asarray(destinations)
            device_physical = xp.asarray(physical)
            order = xp.argsort(device_destinations)
            ordered = device_destinations[order]
            positions = device_physical[order]
            cuts = xp.searchsorted(ordered, xp.asarray(np.r_[0, stops]))
            duplicate = xp.any(ordered[1:] == ordered[:-1])
            summary = xp.concatenate(
                (cuts, duplicate.reshape(1).astype(cuts.dtype))
            )
            host_summary = xp.asnumpy(summary)
            if host_summary[-1]:
                raise ValueError("duplicate VDIF destination slot")
            bounds = xp.empty((len(stops), 2), dtype=xp.int64)
            bounds[:, 0] = -1
            bounds[:, 1] = -1
            for frontier in range(len(stops)):
                lo, hi = map(int, host_summary[frontier : frontier + 2])
                if lo < hi:
                    bounds[frontier, 0] = positions[lo:hi].min()
                    bounds[frontier, 1] = positions[lo:hi].max() + 1
            result = xp.asnumpy(bounds)
        diagnostics["gpu_planned_mappings"] = (
            diagnostics.get("gpu_planned_mappings", 0) + 1
        )
        return result
    except (MemoryError, xp.cuda.memory.OutOfMemoryError):
        # Retire any queued accesses before the temporary arrays are released.
        xp.cuda.get_current_stream().synchronize()
        fallbacks = diagnostics.setdefault("gpu_planner_fallbacks", {})
        fallbacks["allocation-failed"] = (
            fallbacks.get("allocation-failed", 0) + 1
        )
        return None
    except (ImportError, NotImplementedError):
        xp.cuda.get_current_stream().synchronize()
        fallbacks = diagnostics.setdefault("gpu_planner_fallbacks", {})
        fallbacks["unsupported-sort"] = (
            fallbacks.get("unsupported-sort", 0) + 1
        )
        return None
