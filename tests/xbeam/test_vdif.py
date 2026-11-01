# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from decimal import Decimal
from io import BytesIO

import numpy as np
import pytest

from cuphoton.xbeam.adapters.vdif.vdif import (
    DuplicatePolicy,
    DuplicateVdifFrameError,
    TruncatedVdifFrameError,
    UnsupportedVdifError,
    VdifError,
    VdifFrame,
    VdifHeader,
    absolute_frame_slot,
    build_thread_timelines,
    decode_2bit,
    decode_frame,
    iter_vdif_frames,
    mjd_seconds_to_vdif_seconds,
    seconds_since_vdif_origin,
    vdif_seconds_to_mjd_seconds,
)


def make_frame(
    *,
    payload: bytes = bytes([0xE4]) * 8,
    invalid: bool = False,
    legacy: bool = False,
    seconds: int = 7,
    epoch: int = 40,
    frame_number: int = 0,
    version: int = 0,
    channels: int = 1,
    complex_data: bool = False,
    bits_per_sample: int = 2,
    thread_id: int = 3,
    station: str = "AB",
    edv: int = 0,
) -> bytes:
    assert channels > 0 and channels & (channels - 1) == 0
    assert (32 + len(payload)) % 8 == 0
    log2_channels = channels.bit_length() - 1
    word0 = seconds | (int(legacy) << 30) | (int(invalid) << 31)
    word1 = frame_number | (epoch << 24)
    word2 = (
        ((32 + len(payload)) // 8) | (log2_channels << 24) | (version << 29)
    )
    station_id = (ord(station[0]) << 8) | ord(station[1])
    word3 = (
        station_id
        | (thread_id << 16)
        | ((bits_per_sample - 1) << 26)
        | (int(complex_data) << 31)
    )
    words = np.asarray(
        (word0, word1, word2, word3, edv << 24, 0, 0, 0), dtype="<u4"
    )
    return words.tobytes() + payload


def test_header_decodes_standard_fields() -> None:
    header = VdifHeader.from_bytes(
        make_frame(
            seconds=123,
            epoch=17,
            frame_number=9,
            channels=2,
            complex_data=True,
            thread_id=511,
            station="XY",
            edv=7,
        )
    )

    assert header.seconds_from_epoch == 123
    assert header.reference_epoch == 17
    assert header.frame_number == 9
    assert header.frame_length_bytes == 40
    assert header.channel_count == 2
    assert header.complex_data
    assert header.bits_per_sample == 2
    assert header.thread_id == 511
    assert header.station_code == "XY"
    assert header.extended_data_version == 7
    assert header.samples_per_frame == 8


def test_rejects_legacy_version_reserved_and_quantization() -> None:
    with pytest.raises(UnsupportedVdifError, match="legacy"):
        VdifHeader.from_bytes(make_frame(legacy=True))
    with pytest.raises(UnsupportedVdifError, match="version 1"):
        VdifHeader.from_bytes(make_frame(version=1))
    with pytest.raises(UnsupportedVdifError, match="1 bits/sample"):
        VdifHeader.from_bytes(make_frame(bits_per_sample=1))

    data = bytearray(make_frame())
    word1 = int.from_bytes(data[4:8], "little") | (1 << 31)
    data[4:8] = word1.to_bytes(4, "little")
    with pytest.raises(VdifError, match="reserved"):
        VdifHeader.from_bytes(data)
    assert VdifHeader.from_bytes(data, strict=False).frame_number == 0


def test_real_two_bit_codes_are_earliest_first() -> None:
    samples, valid = decode_2bit(bytes((0xE4,)), channels=1)

    np.testing.assert_allclose(
        samples[:, 0], [-3.3359, -1.0, 1.0, 3.3359], rtol=0, atol=1e-6
    )
    np.testing.assert_array_equal(valid, [True, True, True, True])


def test_complex_two_bit_pairs_i_before_q() -> None:
    # Scalar component codes: 0I, 3Q, 1I, 2Q.
    samples, valid = decode_2bit(bytes((0x9C,)), complex_data=True)

    np.testing.assert_allclose(
        samples[:, 0],
        np.asarray((-3.3359 + 3.3359j, -1.0 + 1.0j), dtype=np.complex64),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_array_equal(valid, [True, True])


def test_multichannel_codes_are_time_major_with_lsb_channel_first() -> None:
    # time 0: channels [0, 1]; time 1: channels [2, 3]
    samples, _ = decode_2bit(bytes((0xE4,)), channels=2)
    np.testing.assert_allclose(
        samples,
        [[-3.3359, -1.0], [1.0, 3.3359]],
        rtol=0,
        atol=1e-6,
    )


def test_invalid_frame_decodes_to_zero_and_zero_validity() -> None:
    frame = VdifFrame.from_bytes(make_frame(invalid=True))
    decoded = decode_frame(frame)

    np.testing.assert_array_equal(
        decoded.samples, np.zeros((32, 1), np.float32)
    )
    assert not decoded.validity.any()


def test_frame_iterator_tracks_offsets_and_rejects_truncation() -> None:
    first = make_frame(frame_number=0)
    second = make_frame(frame_number=1)
    frames = list(iter_vdif_frames(BytesIO(first + second)))

    assert [frame.offset for frame in frames] == [0, 40]
    assert [frame.header.frame_number for frame in frames] == [0, 1]
    with pytest.raises(TruncatedVdifFrameError, match="payload"):
        list(iter_vdif_frames(BytesIO(first[:-1])))


def test_absolute_slots_include_epoch_boundaries_and_leap_seconds() -> None:
    before = VdifFrame.from_bytes(
        make_frame(epoch=11, seconds=0)
    ).header  # July 2005
    after = VdifFrame.from_bytes(
        make_frame(epoch=12, seconds=0)
    ).header  # Jan 2006

    nominal_half_year = 184 * 86_400
    assert seconds_since_vdif_origin(after) - seconds_since_vdif_origin(
        before
    ) == (nominal_half_year + 1)
    assert (
        absolute_frame_slot(after, 2) == seconds_since_vdif_origin(after) * 2
    )
    with pytest.raises(VdifError, match="outside"):
        absolute_frame_slot(
            VdifFrame.from_bytes(make_frame(frame_number=2)).header,
            2,
        )


def test_timelines_reorder_fill_gaps_and_preserve_thread_validity() -> None:
    frames = [
        VdifFrame.from_bytes(make_frame(frame_number=2, thread_id=0)),
        VdifFrame.from_bytes(
            make_frame(frame_number=0, thread_id=1, invalid=True)
        ),
        VdifFrame.from_bytes(make_frame(frame_number=0, thread_id=0)),
        VdifFrame.from_bytes(make_frame(frame_number=2, thread_id=1)),
    ]
    timelines = build_thread_timelines(frames, frames_per_second=4)

    np.testing.assert_array_equal(
        timelines[0].frame_validity, [True, False, True]
    )
    np.testing.assert_array_equal(
        timelines[1].frame_validity, [False, False, True]
    )
    samples0, valid0 = timelines[0].samples()
    samples1, valid1 = timelines[1].samples()
    assert samples0.shape == samples1.shape == (96, 1)
    assert valid0[:32].all() and not valid0[32:64].any() and valid0[64:].all()
    assert not valid1[:64].any() and valid1[64:].all()


def test_duplicate_policy_is_deterministic_and_valid_aware() -> None:
    invalid = VdifFrame.from_bytes(make_frame(invalid=True))
    valid_first = VdifFrame.from_bytes(make_frame(payload=bytes((0x00,)) * 8))
    valid_last = VdifFrame.from_bytes(make_frame(payload=bytes((0xFF,)) * 8))

    first_valid = build_thread_timelines(
        (invalid, valid_first, valid_last),
        frames_per_second=1,
        duplicate_policy=DuplicatePolicy.FIRST_VALID,
    )[3]
    assert first_valid.frames == (valid_first,)

    last_valid = build_thread_timelines(
        (invalid, valid_first, valid_last),
        frames_per_second=1,
        duplicate_policy=DuplicatePolicy.LAST_VALID,
    )[3]
    assert last_valid.frames == (valid_last,)

    with pytest.raises(DuplicateVdifFrameError):
        build_thread_timelines(
            (invalid, valid_first),
            frames_per_second=1,
            duplicate_policy=DuplicatePolicy.REJECT,
        )


def test_thread_format_changes_are_rejected() -> None:
    frames = (
        VdifFrame.from_bytes(make_frame(channels=1)),
        VdifFrame.from_bytes(make_frame(channels=2, frame_number=1)),
    )
    with pytest.raises(VdifError, match="inconsistent"):
        build_thread_timelines(frames, frames_per_second=2)


@pytest.mark.parametrize("mjd,leaps", [(53736, 1), (57754, 5)])
def test_utc_vdif_conversion_at_leap_transition(mjd, leaps):
    utc = Decimal(mjd * 86400)
    boundary = Decimal((mjd - 51544) * 86400 + leaps)
    before = utc - Decimal("0.25")
    after = utc + Decimal("0.25")
    assert mjd_seconds_to_vdif_seconds(before) == boundary - Decimal("1.25")
    assert mjd_seconds_to_vdif_seconds(after) == boundary + Decimal("0.25")
    assert vdif_seconds_to_mjd_seconds(boundary - Decimal("1.25")) == before
    assert vdif_seconds_to_mjd_seconds(boundary + Decimal("0.25")) == after
    with pytest.raises(VdifError, match="no normalized MJDTime"):
        vdif_seconds_to_mjd_seconds(boundary - Decimal("0.5"))
