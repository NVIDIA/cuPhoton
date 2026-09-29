# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Importable CPU worker and CUDA stubs for real spawn lifecycle tests."""

from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from cuphoton.core.bulk import atomic_write_json


class Worker:
    def __init__(self, options):
        self.options = options
        self.owner = threading.get_ident()
        self.identifier = f"{os.getpid()}-{threading.get_native_id()}"
        self.trace = Path(options["trace"]) / f"{self.identifier}.txt"
        self.record("initialize")
        if options.get("failure") == "initialize":
            raise ValueError("initialization failed")
        self.gpu_identity = {
            "backend": "cupy",
            "device_index": 0,
            "uuid": "GPU-shared",
            "identity_error": None,
        }

    def record(self, message):
        assert self.owner == threading.get_ident()
        with self.trace.open("a") as stream:
            stream.write(message + "\n")

    def run_item(self, item, output_dir):
        self.record(f"item:{item.item_id}")
        time.sleep(self.options.get("item_delay", 0.0))
        if self.options.get("failure") == "hang":
            time.sleep(30)
        if self.options.get("failure") == "exit":
            os._exit(7)
        if self.options.get("failure") == "item":
            raise ValueError("item failed")
        output_dir.mkdir()
        atomic_write_json(output_dir / "summary.json", item.payload)
        return {
            "run_dir": str(output_dir),
            "summary_path": str(output_dir / "summary.json"),
            "backend": "cupy",
            "device": "cuda:0",
            "runtime": {"instance": self.identifier},
            "timings_sec": {"work": 0.0},
            "wall_sec": {"item": 0.0},
        }

    def close(self):
        self.record("close")
        if self.options.get("failure") == "close":
            raise ValueError("close failed")


def factory(options):
    return Worker(options)


def process_entry(connections, payload, worker_ids, threaded):
    from cuphoton.xscan import sharing

    count = payload["options"].get("device_count", 1)
    pool = SimpleNamespace(
        malloc=lambda size: None, free_all_blocks=lambda: None
    )
    cp = SimpleNamespace(
        cuda=SimpleNamespace(
            runtime=SimpleNamespace(getDeviceCount=lambda: count),
            Device=lambda index: SimpleNamespace(use=lambda: None),
            MemoryPool=lambda: pool,
            using_allocator=lambda allocator: nullcontext(),
        )
    )
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            device_count=lambda: count,
            set_device=lambda index: None,
        )
    )

    def require_mps_client(pid):
        if payload["options"].get("failure") == "mps":
            raise RuntimeError("MPS client verification failed")
        return {
            "client_pid": pid,
            "server_pid": 41,
            "pipe_directory": os.environ["CUDA_MPS_PIPE_DIRECTORY"],
        }

    sys.modules["cuphoton.core.mps"] = SimpleNamespace(
        require_mps_client=require_mps_client
    )
    sys.modules["cupy"] = cp
    sys.modules["torch"] = torch
    sharing._process_entry(connections, payload, worker_ids, threaded)
