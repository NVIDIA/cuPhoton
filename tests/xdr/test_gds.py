# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import ModuleType

import pytest

from cuphoton.xdr import gds


@pytest.fixture
def kvikio_defaults(monkeypatch):
    kvikio = ModuleType("kvikio")
    defaults = ModuleType("kvikio.defaults")
    kvikio.defaults = defaults
    monkeypatch.setitem(sys.modules, "kvikio", kvikio)
    monkeypatch.setitem(sys.modules, "kvikio.defaults", defaults)
    monkeypatch.setattr(gds, "_KVIKIO_NUM_THREADS", None)
    monkeypatch.delenv("KVIKIO_NTHREADS", raising=False)
    return defaults


def test_concurrent_readers_initialize_pool_once(
    monkeypatch, kvikio_defaults
):
    pool_threads = 1
    resets = []
    start = Barrier(8)

    def reset_pool(name, count):
        resets.append((name, count))
        # The real C++ pool reset releases the GIL. Other readers must wait
        # until it finishes, then use this pool without resetting it again.
        time.sleep(0.01)
        nonlocal pool_threads
        pool_threads = count

    def configure_reader():
        start.wait(timeout=5)
        return gds.configure_kvikio_parallelism()

    monkeypatch.setattr(gds, "available_cpu_cores", lambda: 12)
    kvikio_defaults.set = reset_pool
    kvikio_defaults.get = lambda name: pool_threads
    with ThreadPoolExecutor(max_workers=8) as readers:
        futures = [readers.submit(configure_reader) for _ in range(8)]
        assert [future.result(timeout=5) for future in futures] == [12] * 8
    assert resets == [("num_threads", 12)]

    # Later readers cannot resize a pool that earlier readers still use.
    monkeypatch.setattr(gds, "available_cpu_cores", lambda: 24)
    assert gds.configure_kvikio_parallelism() == 12
    assert resets == [("num_threads", 12)]


def test_explicit_kvikio_threads_do_not_reset_pool(
    monkeypatch, kvikio_defaults
):
    monkeypatch.setenv("KVIKIO_NTHREADS", "2")
    kvikio_defaults.get = lambda name: 2

    def reject_reset(*args):
        raise AssertionError("Explicit KvikIO configuration was overwritten")

    kvikio_defaults.set = reject_reset
    monkeypatch.setattr(gds, "available_cpu_cores", reject_reset)

    assert gds.configure_kvikio_parallelism() == 2
    assert gds.configure_kvikio_parallelism() == 2


def test_failed_pool_initialization_can_retry(monkeypatch, kvikio_defaults):
    attempts = []

    def reset_pool(name, count):
        attempts.append(count)
        if len(attempts) == 1:
            raise RuntimeError("Pool initialization failed")

    monkeypatch.setattr(gds, "available_cpu_cores", lambda: 4)
    kvikio_defaults.set = reset_pool
    kvikio_defaults.get = lambda name: 4 if len(attempts) >= 2 else 1

    with pytest.raises(RuntimeError, match="Pool initialization failed"):
        gds.configure_kvikio_parallelism()
    assert gds.configure_kvikio_parallelism() == 4
    assert attempts == [4, 4]
