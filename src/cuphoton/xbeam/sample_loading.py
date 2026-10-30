# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Sample loading implementation."""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Any, cast

import numpy as np

from cuphoton.xbeam.adapters.radio.compute_plan import (
    _mjd_absolute_seconds,
    _sample_rate,
    _sample_span_limit,
)
from cuphoton.xbeam.adapters.radio.models import (
    Datastream,
    Frequency,
    Job,
    RecordedBand,
    SamplingKind,
    VoltageBatch,
)
from cuphoton.xbeam.adapters.vdif.recording import VdifPayloadPlan
from cuphoton.xbeam.adapters.vdif.vdif import decode_2bit

from .errors import ConfigurationError
from .runtime_types import _BandSamples
from .upload_resources import (
    load_vdif_payload_plans,
)


def _load_band_samples(
    job: Job,
    datastream: Datastream,
    band: RecordedBand,
    frequency: Frequency,
    batches: list[VoltageBatch],
    xp: Any = np,
    device_cache: dict[tuple[str, int], tuple[Any, Any]] | None = None,
    defer_file_payload: bool = False,
    reservation_token: object | None = None,
    planned_sample_coverage: bool = True,
) -> _BandSamples:
    sample_rate = _sample_rate(datastream, frequency)
    entity = (
        f"datastream[{datastream.identifier}]"
        f".recorded_band[{band.identifier}]"
    )
    if not batches:
        raise ConfigurationError(
            entity,
            "band_id",
            "recording source yielded no batch for this band",
        )

    job_time = _mjd_absolute_seconds(
        job.start_time.mjd, job.start_time.seconds_of_day
    )
    samples: np.ndarray | VdifPayloadPlan
    segments = []
    packed_layout = batches[0].packed_layout
    for batch_index, batch in enumerate(batches):
        if not math.isclose(
            batch.sample_rate_hz, sample_rate, rel_tol=1e-12, abs_tol=1e-6
        ):
            raise ConfigurationError(
                entity,
                f"batch[{batch_index}].sample_rate_hz",
                f"expected {sample_rate}, got {batch.sample_rate_hz}",
            )
        if batch.polarization != band.polarization:
            raise ConfigurationError(
                entity,
                f"batch[{batch_index}].polarization",
                f"expected {band.polarization!r}, got {batch.polarization!r}",
            )
        if batch.packed_layout != packed_layout:
            raise ConfigurationError(
                entity,
                f"batch[{batch_index}].packed_layout",
                "all batches for one band must use the same representation",
            )
        first_time = _mjd_absolute_seconds(
            batch.start_time.mjd, batch.start_time.seconds_of_day
        )
        sample_position = (first_time - job_time) * Decimal(
            str(batch.sample_rate_hz)
        )
        integral_position = sample_position.to_integral_value()
        if sample_position != integral_position:
            raise ConfigurationError(
                entity,
                f"batch[{batch_index}].start_time",
                "must fall on the declared sample grid",
            )
        segment_start = int(integral_position)
        if batch.packed_layout is None:
            decoded = np.asarray(batch.samples)
            expects_complex = datastream.sampling is not SamplingKind.REAL
            dtype_matches_sampling = (
                np.iscomplexobj(decoded) == expects_complex
            )
            if decoded.dtype.kind not in "iufc" or not dtype_matches_sampling:
                raise ConfigurationError(
                    entity,
                    f"batch[{batch_index}].samples",
                    "decoded sample dtype does not match datastream sampling",
                )
            samples = np.ascontiguousarray(
                decoded.reshape(-1),
                dtype=np.complex64 if expects_complex else np.float32,
            )
            validity = np.ascontiguousarray(
                np.asarray(batch.validity, dtype=np.bool_).reshape(-1)
            )
            if validity.size != samples.size:
                raise ConfigurationError(
                    entity,
                    f"batch[{batch_index}].validity",
                    "decoded samples require one validity value per sample",
                )
            sample_count = samples.size
        else:
            layout = batch.packed_layout
            file_plan = (
                batch.samples
                if isinstance(batch.samples, VdifPayloadPlan)
                else None
            )
            if file_plan is None:
                samples = np.ascontiguousarray(
                    np.asarray(batch.samples, dtype=np.uint8).reshape(-1)
                )
            else:
                samples = file_plan
            if (
                layout.frame_payload_bytes <= 0
                or layout.samples_per_frame <= 0
            ):
                raise ConfigurationError(
                    entity,
                    f"batch[{batch_index}].packed_layout",
                    "packed frame sizes must be positive",
                )
            if layout.channel_count <= 0 or layout.channel_count & (
                layout.channel_count - 1
            ):
                raise ConfigurationError(
                    entity,
                    f"batch[{batch_index}].packed_layout.channel_count",
                    "must be a positive power of two",
                )
            if (
                file_plan is not None
                and file_plan.payload_bytes != layout.frame_payload_bytes
            ):
                raise ConfigurationError(
                    entity,
                    f"batch[{batch_index}].samples",
                    "VDIF payload plan does not match the packed layout",
                )
            components_per_sample = layout.channel_count * (
                2 if layout.complex_data else 1
            )
            if (
                layout.frame_payload_bytes * 4
                != layout.samples_per_frame * components_per_sample
            ):
                raise ConfigurationError(
                    entity,
                    f"batch[{batch_index}].packed_layout.samples_per_frame",
                    "does not match the two-bit payload layout",
                )
            packed_byte_count = (
                samples.byte_count
                if isinstance(samples, VdifPayloadPlan)
                else samples.size
            )
            if packed_byte_count % layout.frame_payload_bytes:
                raise ConfigurationError(
                    entity,
                    f"batch[{batch_index}].samples",
                    "packed payload size is not an integral number of frames",
                )
            frame_count = packed_byte_count // layout.frame_payload_bytes
            validity = np.ascontiguousarray(
                np.asarray(batch.validity, dtype=np.bool_).reshape(-1)
            )
            if validity.size != frame_count:
                raise ConfigurationError(
                    entity,
                    f"batch[{batch_index}].validity",
                    "packed samples require one validity value per frame",
                )
            if layout.complex_data != (
                datastream.sampling is not SamplingKind.REAL
            ):
                raise ConfigurationError(
                    entity,
                    f"batch[{batch_index}].packed_layout.complex_data",
                    "does not match datastream sampling",
                )
            if not 0 <= layout.channel_index < layout.channel_count:
                raise ConfigurationError(
                    entity,
                    f"batch[{batch_index}].packed_layout.channel_index",
                    "must select an available packed channel",
                )
            sample_count = frame_count * layout.samples_per_frame
        segments.append((segment_start, sample_count, samples, validity))

    segments.sort(key=lambda item: item[0])
    start_sample = segments[0][0]
    stop_sample = max(
        start + count for start, count, _samples, _validity in segments
    )
    previous_stop = start_sample
    for segment_start, sample_count, _samples, _validity in segments:
        if segment_start < previous_stop:
            raise ConfigurationError(
                entity, "batches", "recording batches overlap"
            )
        previous_stop = segment_start + sample_count

    sample_span = stop_sample - start_sample
    sample_limit = _sample_span_limit(
        job,
        sample_rate,
        sum(count for _start, count, _samples, _validity in segments),
    )
    if sample_span > sample_limit:
        raise ConfigurationError(
            entity,
            "batches",
            "batch timestamp span exceeds the declared job interval and "
            "observed sample count",
        )

    if packed_layout is None:
        dtype = np.result_type(
            *(cast(np.ndarray, item[2]).dtype for item in segments)
        )
        values = np.zeros(stop_sample - start_sample, dtype=dtype)
        validity = np.zeros(stop_sample - start_sample, dtype=np.bool_)
        for (
            segment_start,
            sample_count,
            samples,
            segment_validity,
        ) in segments:
            offset = segment_start - start_sample
            values[offset : offset + sample_count] = samples
            validity[offset : offset + sample_count] = segment_validity
        device_payload = None
        validity_stride = 1
        source_sample_count = None
        packed_samples_per_frame = 0
        packed_payload_bytes = 0
        packed_channel_count = 0
        packed_channel_index = 0
    else:
        layout = packed_layout
        total_samples = stop_sample - start_sample
        if total_samples % layout.samples_per_frame:
            raise ConfigurationError(
                entity,
                "batches",
                "packed batch gaps must align to frame boundaries",
            )
        frame_count = total_samples // layout.samples_per_frame
        if len(segments) == 1:
            # The common file-backed path is already contiguous.  Keep the
            # shared buffers intact: avoiding a second host copy is important
            # for large recordings and preserves one GPU upload per thread.
            packed = segments[0][2]
            frame_validity = segments[0][3]
            payload_cache_host = batches[0].samples
            validity_cache_host = batches[0].validity
        else:
            packed = np.zeros(
                frame_count * layout.frame_payload_bytes, dtype=np.uint8
            )
            frame_validity = np.zeros(frame_count, dtype=np.bool_)
            for (
                segment_start,
                sample_count,
                samples,
                segment_validity,
            ) in segments:
                sample_offset = segment_start - start_sample
                if sample_offset % layout.samples_per_frame:
                    raise ConfigurationError(
                        entity,
                        "batches",
                        "packed batches must begin on frame boundaries",
                    )
                frame_offset = sample_offset // layout.samples_per_frame
                segment_frames = sample_count // layout.samples_per_frame
                byte_offset = frame_offset * layout.frame_payload_bytes
                segment_payload = (
                    samples.materialize()
                    if isinstance(samples, VdifPayloadPlan)
                    else samples
                )
                packed[byte_offset : byte_offset + segment_payload.size] = (
                    segment_payload
                )
                frame_validity[
                    frame_offset : frame_offset + segment_frames
                ] = segment_validity
            payload_cache_host = packed
            validity_cache_host = frame_validity
        payload_cache_token = id(payload_cache_host)
        validity_cache_token = id(validity_cache_host)
        validity_stride = layout.samples_per_frame
        source_sample_count = frame_count * layout.samples_per_frame
        packed_samples_per_frame = layout.samples_per_frame
        packed_payload_bytes = layout.frame_payload_bytes
        packed_channel_count = layout.channel_count
        packed_channel_index = layout.channel_index
        if xp.__name__ == "cupy":
            values = np.empty(
                0,
                dtype=np.complex64 if layout.complex_data else np.float32,
            )
            validity = np.empty(0, dtype=np.bool_)
            device_payload = None
            if device_cache is not None:
                cached_payload = device_cache.get(
                    ("samples", payload_cache_token)
                )
                if (
                    cached_payload is not None
                    and cached_payload[0] is payload_cache_host
                ):
                    device_payload = cached_payload[1]
            if device_payload is None and not isinstance(
                packed, VdifPayloadPlan
            ):
                device_payload = xp.asarray(packed)
                if device_cache is not None:
                    device_cache[("samples", payload_cache_token)] = (
                        payload_cache_host,
                        device_payload,
                    )
        else:
            if isinstance(packed, VdifPayloadPlan):
                cached_host_payload = None
                if device_cache is not None:
                    cached_host_payload = device_cache.get(
                        ("host_samples", payload_cache_token)
                    )
                if (
                    cached_host_payload is not None
                    and cached_host_payload[0] is payload_cache_host
                ):
                    packed = cached_host_payload[1]
                else:
                    packed = packed.materialize()
                    if device_cache is not None:
                        device_cache[
                            ("host_samples", payload_cache_token)
                        ] = (
                            payload_cache_host,
                            packed,
                        )
            decoded, _ignored = decode_2bit(
                packed,
                complex_data=layout.complex_data,
                channels=layout.channel_count,
                xp=np,
            )
            values = np.ascontiguousarray(decoded[:, layout.channel_index])
            validity = np.repeat(frame_validity, layout.samples_per_frame)
            device_payload = None

    device_values = None
    device_validity = None
    if xp.__name__ == "cupy":
        if packed_layout is None:
            device_values = xp.asarray(values)
            device_validity = xp.asarray(validity, dtype=xp.uint8)
        else:
            if device_cache is not None:
                cached_validity = device_cache.get(
                    ("validity", validity_cache_token)
                )
                if (
                    cached_validity is not None
                    and cached_validity[0] is validity_cache_host
                ):
                    device_validity = cached_validity[1]
            if device_validity is None:
                streaming_payload = (
                    isinstance(packed, VdifPayloadPlan)
                    and packed.streaming is not None
                )
                if streaming_payload:
                    # The uploader derives validity from resident headers.
                    # Do not enqueue a pointless asynchronous H2D copy and
                    # then mutate its NumPy source while that copy is pending.
                    frame_validity.fill(False)
                    device_validity = xp.zeros(
                        frame_validity.shape, dtype=xp.uint8
                    )
                else:
                    device_validity = xp.asarray(
                        frame_validity, dtype=xp.uint8
                    )
                if device_cache is not None:
                    device_cache[("validity", validity_cache_token)] = (
                        validity_cache_host,
                        device_validity,
                    )

            if (
                device_payload is None
                and isinstance(packed, VdifPayloadPlan)
                and not defer_file_payload
            ):
                device_payload = load_vdif_payload_plans(
                    [packed],
                    xp=xp,
                    validity_outputs=[device_validity]
                    if packed.streaming is not None
                    else None,
                    reservation_token=reservation_token,
                )[0]
                if device_cache is not None:
                    device_cache[("samples", payload_cache_token)] = (
                        payload_cache_host,
                        device_payload,
                    )

    return _BandSamples(
        datastream=datastream,
        band=band,
        frequency=frequency,
        samples=values,
        validity=np.asarray(validity),
        start_sample=start_sample,
        sample_rate_hz=sample_rate,
        planned_sample_coverage=planned_sample_coverage,
        device_samples=device_values,
        device_validity=device_validity,
        validity_stride=validity_stride,
        device_payload=device_payload,
        source_sample_count=source_sample_count,
        packed_samples_per_frame=packed_samples_per_frame,
        packed_payload_bytes=packed_payload_bytes,
        packed_channel_count=packed_channel_count,
        packed_channel_index=packed_channel_index,
        payload_plan=(
            packed
            if packed_layout is not None
            and isinstance(packed, VdifPayloadPlan)
            else None
        ),
        payload_frame_validity=(
            np.asarray(frame_validity) if packed_layout is not None else None
        ),
    )
