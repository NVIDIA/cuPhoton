# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Physical presentation uses global file IDs and local bounded range IDs."""

from pathlib import Path

import numpy as np
import pytest

from cuphoton.xbeam.adapters.radio.models import (
    Datastream,
    RecordedBand,
    SamplingKind,
)
from cuphoton.xbeam.adapters.vdif.recording import (
    VdifFileRecordingSource,
    VdifSourceRange,
    VdifThreadWindow,
)
from cuphoton.xbeam.adapters.vdif.vdif import VdifHeader, absolute_frame_slot
from cuphoton.xbeam.adapters.vdif.vdif_sidecar_builder import (
    build_vdif_binary_index,
)
from cuphoton.xbeam.profiles import ANALYTIC_V1
from tests.xbeam.test_recording import _frame, _job


@pytest.mark.parametrize("ranges", [((0, 3),), ((0, 1), (2, 3)), ((1, 3),)])
def test_global_file_one_with_only_local_source_ranges(
    tmp_path: Path, ranges
):
    decoy = tmp_path / "file-zero.vdif"
    decoy.write_bytes(_frame(payload=bytes([0xEE]) * 8, frame_number=0))
    path = tmp_path / "file-one.vdif"
    payloads = [bytes([0x10 + frame]) * 8 for frame in range(3)]
    path.write_bytes(
        b"".join(
            _frame(
                payload=payload,
                frame_number=frame,
                seconds_from_epoch=10,
                invalid=frame == 1,
            )
            for frame, payload in enumerate(payloads)
        )
    )
    stream = Datastream(
        identifier=0,
        station_id=0,
        sampling=SamplingKind.REAL,
        quantization_bits=2,
        frame_size_bytes=40,
        source_files=(str(decoy), str(path)),
        recorded_bands=(RecordedBand(0, 0, "R", thread_id=5),),
        sample_rate_hz=64.0,
    )
    # Destination headers define a job clock independent of native file1 time.
    template = VdifHeader.from_bytes(_frame(payload=bytes(8), frame_number=0))
    job_start = absolute_frame_slot(template, 4)
    first, stop = (
        min(left for left, _ in ranges),
        max(right for _, right in ranges),
    )
    windows = {
        0: {
            5: VdifThreadWindow(
                template, 4, job_start + first, job_start + stop
            )
        }
    }
    stat = path.stat()
    requested = tuple(
        VdifSourceRange(
            path,
            left,
            right - left,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_dev,
            stat.st_ino,
        )
        for left, right in ranges
    )
    indexes = tmp_path / "indexes"
    build_vdif_binary_index(path, index_dir=indexes, backend="posix")
    (batch,) = list(
        VdifFileRecordingSource(
            _job(stream, 64.0),
            ANALYTIC_V1,
            source_ranges={0: requested},
            thread_windows=windows,
            source_thread_windows=windows,
            binary_sidecars=True,
            binary_index_dir=indexes,
            physical_frame_presentation_plans={0: ((1, 0, 3, job_start),)},
        ).batches(stream)
    )
    selected = {
        frame for left, right in ranges for frame in range(left, right)
    }
    expected = b"".join(
        payloads[frame] if frame in selected else bytes(8)
        for frame in range(first, stop)
    )
    np.testing.assert_array_equal(
        np.asarray(batch.samples), np.frombuffer(expected, dtype=np.uint8)
    )
    np.testing.assert_array_equal(
        batch.validity,
        [frame in selected and frame != 1 for frame in range(first, stop)],
    )
    assert len(batch.samples.files) == len(ranges)
    assert all(file.path == path for file in batch.samples.files)
