# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.xbeam import ingest_scheduler as scheduler
from cuphoton.xbeam.adapters.vdif.recording import (
    VdifPayloadFilePlan,
    VdifPayloadPlan,
)


def schedule(mappings, slots, frontiers=4, diagnostics=None):
    plan = VdifPayloadPlan(
        40,
        8,
        slots,
        tuple(
            VdifPayloadFilePlan(
                Path(f"{i}.vdif"),
                len(mapping),
                np.asarray(mapping, np.int32),
                first_frame=7,
            )
            for i, mapping in enumerate(mappings)
        ),
    )
    validity = np.zeros(slots, np.uint8)
    for mapping in mappings:
        mapping = np.asarray(mapping)
        validity[mapping[mapping >= 0]] = 1
    output = np.zeros(plan.byte_count, np.uint8)
    progress = scheduler._PayloadProgress(slots, static_validity=validity)
    result = scheduler._mapped_frontier_upload_tasks(
        [[plan]],
        {id(plan): output},
        {id(plan): validity},
        {id(plan): progress},
        [440],
        frontiers,
        trust_slices=True,
        diagnostics=diagnostics,
    )
    return result, progress, validity


@pytest.mark.parametrize(
    "distance,eligible", [(9, True), (10, True), (11, False)]
)
def test_read_amplification_boundary(distance, eligible, monkeypatch):
    mapping = np.arange(400)
    mapping[200 - distance], mapping[199 + distance] = (
        mapping[199 + distance],
        mapping[200 - distance],
    )
    info = {}
    if not eligible:

        def fail(*args):
            pytest.fail("rejected plan allocated completion futures")

        monkeypatch.setattr(scheduler, "_FrontierPlanCompletion", fail)
    result, _, _ = schedule([mapping], 400, diagnostics=info)
    assert (result is not None) == eligible
    assert info["mapping_order"] == "unordered"
    assert info["read_amplification"] == (400 + 2 * distance) / 400
    assert info["fallback_detail"] == (
        None if eligible else "excessive-read-amplification"
    )


def test_widely_shuffled_frames_fall_back():
    result, _, _ = schedule([np.random.default_rng(4).permutation(400)], 400)
    assert result is None


@pytest.mark.parametrize("mappings", [([2, 0, 2, 1],), ([2, 0, 1], [3, 2])])
def test_nonadjacent_and_cross_file_duplicates_rejected(mappings):
    with pytest.raises(ValueError, match="duplicate"):
        schedule(mappings, 4)


def test_physical_overlaps_do_not_publish_neighbor_or_delayed_frames():
    mapping = np.arange(400)
    mapping[99], mapping[100] = mapping[100], mapping[99]
    mapping[5] = -1  # Invalid physical entry; logical slot 5 is missing.
    (tasks, futures), progress, validity = schedule([mapping], 400)
    first, second = tasks[:2]
    m0 = first.plans[0].files[0].destination_slots
    m1 = second.plans[0].files[0].destination_slots
    assert m0[-2] == -1  # Slot 100 belongs exclusively to the next frontier.
    assert m1[1] == -1  # Slot 99 belongs exclusively to the first frontier.
    first.progresses[0].publish(m0[:-1], object())
    assert not progress.samples_published(0, 100, 1, validity)
    second.progresses[0].publish(m1, object())
    assert not progress.samples_published(0, 100, 1, validity)
    event = object()
    first.progresses[0].publish(m0[-1:], event)
    assert progress.samples_published(0, 100, 1, validity)
    assert event in progress._events.values()
    waited = []
    xp = SimpleNamespace(
        cuda=SimpleNamespace(
            get_current_stream=lambda: SimpleNamespace(
                wait_event=waited.append
            )
        )
    )
    progress.wait_samples(0, 100, 1, validity, xp)
    assert event in waited
    assert not any(f.done() for f in futures.values())
    for task in tasks:
        task.progresses[0].finish(None)
    assert all(f.done() for f in futures.values())


def test_multifile_missing_frames_and_empty_frontiers():
    a = np.r_[np.arange(100), [-1] * 10]
    a[20], a[21] = a[21], a[20]
    b = np.arange(300, 400)
    (tasks, _), _, _ = schedule([a, b], 400)
    assert not tasks[1].plans[0].files
    assert not tasks[2].plans[0].files
    seen = []
    for frontier, task in enumerate(tasks):
        for file in task.plans[0].files:
            selected = file.destination_slots
            seen.extend((selected[selected >= 0] + 100 * frontier).tolist())
    assert sorted(seen) == list(range(100)) + list(range(300, 400))


def test_shared_read_plans_union_bounds_without_cross_frontier_writes():
    mappings = [np.arange(400, dtype=np.int32) for _ in range(2)]
    mappings[0][99], mappings[0][100] = mappings[0][100], mappings[0][99]
    mappings[1][98], mappings[1][101] = mappings[1][101], mappings[1][98]
    plans = [
        VdifPayloadPlan(
            40,
            8,
            400,
            (VdifPayloadFilePlan(Path("shared.vdif"), 400, mapping),),
        )
        for mapping in mappings
    ]
    outputs = {
        id(plan): np.zeros(plan.byte_count, np.uint8) for plan in plans
    }
    validity = {id(plan): np.ones(400, np.uint8) for plan in plans}
    progresses = {id(plan): scheduler._PayloadProgress(400) for plan in plans}
    tasks, _ = scheduler._mapped_frontier_upload_tasks(
        [plans],
        outputs,
        validity,
        progresses,
        [440],
        4,
        trust_slices=True,
    )
    seen = [[], []]
    for frontier, task in enumerate(tasks):
        files = [plan.files[0] for plan in task.plans]
        assert len({(f.first_frame, f.frame_count) for f in files}) == 1
        for index, file in enumerate(files):
            for physical, destination in enumerate(file.destination_slots):
                if destination >= 0:
                    parent = mappings[index][file.first_frame + physical]
                    assert parent == 100 * frontier + destination
                    seen[index].append(int(parent))
    assert all(sorted(values) == list(range(400)) for values in seen)
