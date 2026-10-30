# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Buffered reader requests: input-derived compatibility planning."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

import numpy as np

from cuphoton.xbeam.adapters.radio.compute_plan import (
    _delay_values,
    _ordered_model_runs,
    _sample_rate,
    _source_block_indices_for_scan,
)
from cuphoton.xbeam.adapters.radio.models import (
    Datastream,
    Frequency,
    Job,
    RecordedBand,
    Scan,
)

from ._checks import _integer, _readonly
from .clocks import (
    LeadingRawHeadContract,
    LeadingRawHeadInputs,
    compile_leading_raw_head_coordinates,
    compile_raw_window_bounds,
)
from .geometry import (
    BufferedNormalFullExtent,
    BufferedRequestGeometry,
    _cursor_add,
    _cursor_multiply,
    _cursor_subtract,
    buffered_request_cursors,
    compile_buffered_request_discards,
    compile_latched_raw_window_heads,
    sample_offset_integer,
)
from .replay import (
    RawReplayWindowFacts,
    RawWindowClockContract,
    RetainedRawWindowContract,
    apply_uncovered_head_publication,
    compile_raw_replay_window_facts,
)
from .validity import RawValidityState


class ReadClockConvention(StrEnum):
    """Declared payload-read clock, separate from GPU timing."""

    ENDPOINT_MEAN = "endpoint-mean"


def _endpoint_model_delays(
    job: Job,
    datastream: Datastream,
    band: RecordedBand,
    scan: Scan,
    times: np.ndarray,
) -> np.ndarray:
    """Select the active model independently at each literal endpoint."""
    delays = np.empty(times.shape, dtype=np.float64)
    for model, selected in _ordered_model_runs(
        scan, datastream.station_id, times
    ):
        delays[selected] = _delay_values(
            job, model, datastream.station_id, times[selected], band, np
        )
    if not np.all(np.isfinite(delays)):
        raise ValueError("read endpoint model delays must be finite")
    return delays


@dataclass(frozen=True, slots=True)
class RawJobRequestInputs:
    """Immutable request clocks in the common job-relative sample basis.

    Counts clip the final timeline request. Endpoint clocks still use its
    nominal complete duration; scope clipping follows cursor compilation.
    """

    first_fft: int
    stop_fft: int
    timeline_fft_count: int
    timestamp_origin_frame: int
    geometry: BufferedRequestGeometry
    request_fft_starts: np.ndarray
    request_fft_counts: np.ndarray
    desired_sample_starts: np.ndarray
    endpoint_drifts: np.ndarray
    ordinary_source_starts: np.ndarray
    sample_rate: float
    leading_head_inputs: LeadingRawHeadInputs | None = None
    read_clock_convention: ReadClockConvention | None = None
    centre_delays_seconds: np.ndarray | None = None
    output_sample_starts: np.ndarray | None = None
    read_average_delays_seconds: np.ndarray | None = None

    def __post_init__(self) -> None:
        for name in (
            "request_fft_starts",
            "request_fft_counts",
            "desired_sample_starts",
            "endpoint_drifts",
            "ordinary_source_starts",
        ):
            object.__setattr__(self, name, _readonly(getattr(self, name)))
        optional = (
            "centre_delays_seconds",
            "output_sample_starts",
            "read_average_delays_seconds",
        )
        if self.read_clock_convention is None:
            if any(getattr(self, name) is not None for name in optional):
                raise ValueError("read clocks require an explicit convention")
        elif self.read_clock_convention is ReadClockConvention.ENDPOINT_MEAN:
            for name in optional:
                array = getattr(self, name)
                if (
                    not isinstance(array, np.ndarray)
                    or array.shape != self.ordinary_source_starts.shape
                ):
                    raise ValueError(
                        "read clock matrices must match ordinary shape"
                    )
                dtype = (
                    np.int64 if name == "output_sample_starts" else np.float64
                )
                if array.dtype != dtype or not np.all(np.isfinite(array)):
                    raise ValueError(
                        "read clock matrices need finite typed values"
                    )
                object.__setattr__(self, name, _readonly(array))
        else:
            raise ValueError("unsupported read clock convention")

    def compile(
        self,
        validity_state: RawValidityState,
        *,
        frame_bytes: int,
        control_word_bits: int,
        window_clock_contract: RetainedRawWindowContract,
    ) -> RawReplayWindowFacts:
        """Compile planner facts with an explicit retained returned view.

        Optional endpoint-mean matrices are retained for explicit consumers;
        they do not select a source policy. A supplied returned contract must
        reach the source compiler before address and publication decisions;
        applying only its window bounds afterward loses its physical heads.
        """
        if (
            validity_state.source_state.request_state.timestamp_origin_frame
            != self.timestamp_origin_frame
        ):
            raise ValueError(
                "raw state and job request timestamp origins differ"
            )
        return compile_raw_replay_window_facts(
            validity_state,
            self.request_fft_starts,
            self.desired_sample_starts,
            self.endpoint_drifts,
            self.ordinary_source_starts,
            self.geometry,
            first_fft=self.first_fft,
            stop_fft=self.stop_fft,
            sample_rate=self.sample_rate,
            frame_bytes=frame_bytes,
            control_word_bits=control_word_bits,
            request_fft_counts=self.request_fft_counts,
            window_clock_contract=window_clock_contract,
        )

    def _normal_full_window_binding(
        self, validity_state, extent, *, control_word_bits
    ):
        if not isinstance(validity_state, RawValidityState):
            raise TypeError("validity state must be RawValidityState")
        if not isinstance(extent, BufferedNormalFullExtent):
            raise TypeError("extent must be BufferedNormalFullExtent")
        if self.read_clock_convention is None:
            raise ValueError("window binding requires explicit model clocks")
        state = validity_state.source_state.request_state
        if state.timestamp_origin_frame != self.timestamp_origin_frame:
            raise ValueError("raw state and request timestamp origins differ")
        if (
            extent.samples_per_byte != self.geometry.samples_per_byte
            or extent.samples_per_frame != state.samples_per_frame
            or extent.request_frames * extent.read_buffer_ratio
            != state.allocation_frames
        ):
            raise ValueError("returned extent and raw geometry differ")
        if extent.framed_admission_samples.denominator != 1:
            raise ValueError("window admission requires integral samples")
        batch = compile_buffered_request_discards(
            state,
            self.request_fft_starts,
            self.desired_sample_starts,
            self.endpoint_drifts,
            self.ordinary_source_starts,
            self.geometry,
            sample_rate=self.sample_rate,
            frame_bytes=extent.frame_bytes,
            control_word_bits=control_word_bits,
            request_fft_counts=self.request_fft_counts,
        )
        heads = compile_latched_raw_window_heads(state, batch, self.geometry)
        columns = np.arange(self.ordinary_source_starts.shape[1])
        indices = self.request_fft_starts[:, None] + columns
        selected = (
            (columns < self.request_fft_counts[:, None])
            & (indices >= self.first_fft)
            & (indices < self.stop_fft)
        )
        if not np.array_equal(
            indices[selected], np.arange(self.first_fft, self.stop_fft)
        ):
            raise ValueError("requests do not cover the window scope")
        size = self.stop_fft - self.first_fft
        contract = RawWindowClockContract(
            self.first_fft,
            self.output_sample_starts[selected],
            np.broadcast_to(heads[:, None], selected.shape)[selected],
            self.centre_delays_seconds[selected],
            np.full(
                size, int(extent.framed_admission_samples), dtype=np.int64
            ),
            np.full(size, extent.decoded_payload_samples, dtype=np.int64),
        )
        return contract, batch, selected

    def published_normal_full_window_contract(
        self,
        validity_state: RawValidityState,
        extent: BufferedNormalFullExtent,
        *,
        control_word_bits: int,
        ordinary_head_contract: LeadingRawHeadContract | None = None,
    ) -> RetainedRawWindowContract:
        """Bind a full return with retained allocation-tail publication.

        The caller establishes normal full returns and this publication
        policy. An event's leading FFT cannot publish from the next physical
        allocation through its retained predecessor view. If that selected
        predecessor starts in uncovered physical frames, its body cannot
        publish such tails either. Explicit-invalid frames retain physical
        coverage; only missing trustworthy source intervals define a hole.
        This also applies the independent uncovered source-head gate.

        For every lane packing, an unchanged retained head also cannot
        publish starts beyond its framed allocation extent when the successor
        anchor advances. Header admission remains distinct from decoded
        payload coverage; tails inside that allowance are preserved.

        An optional ordinary-head contract additionally establishes endpoint
        nanosecond frame selection and accepted-negative decoder clamping for
        non-event request starts. Interior and retained event heads keep their
        own publication contracts. No clock quantum is guessed by default.

        All decisions use request position, source coverage and returned
        cursors. No dataset identity, output coordinates or loss calibration
        selects the policy. Initial, partial and EOF returns still require
        separately established contracts; no execution profile is activated.
        """
        return self._returned_normal_full_window_contract(
            validity_state,
            extent,
            control_word_bits=control_word_bits,
            advance_empty_prefix=True,
            allocation_tail_ownership=True,
            ordinary_head_contract=ordinary_head_contract,
        )

    def _returned_normal_full_window_contract(
        self,
        validity_state,
        extent,
        *,
        control_word_bits,
        advance_empty_prefix,
        allocation_tail_ownership=False,
        ordinary_head_contract=None,
    ):
        clock, batch, selected = self._normal_full_window_binding(
            validity_state, extent, control_word_bits=control_word_bits
        )
        state = validity_state.source_state.request_state
        timeline = validity_state.source_state.timeline
        width = self.ordinary_source_starts.shape[1]
        positions = np.arange(width, dtype=np.int64)
        active = positions[None, :] < self.request_fft_counts[:, None]
        eligible = active & (
            positions[None, :] >= batch.cursors.skipped_ffts[:, None]
        )
        events = batch.event_requests
        if np.any(events):
            framed = (
                _cursor_add(
                    batch.cursors.origin_bytes[events, None],
                    _cursor_multiply(
                        positions[None, :],
                        self.geometry.payload_bytes_per_fft,
                        "retained control FFT offset",
                    ),
                    "retained framed request cursor",
                )
                < extent.allocation_framed_bytes
            )
            framed &= eligible[events]
            framed &= np.any(
                framed[:, : min(width, control_word_bits)], axis=1
            )[:, None]
            eligible[events] = framed
        allocations = batch.selected_allocations.copy()
        cursors = batch.cursors.origin_bytes.copy()
        heads = clock.aligned_head_samples.copy()
        if advance_empty_prefix:
            transfer = events & ~np.any(eligible, axis=1)
            allocations[transfer] += 1
            if np.any(allocations >= state.anchor_samples.size):
                raise ValueError("successor window lies outside raw history")
            if np.any(transfer):
                cursors[transfer] = 0
                request_heads = np.broadcast_to(
                    state.anchor_samples[allocations, None], selected.shape
                )
                transfer_scope = np.broadcast_to(
                    transfer[:, None], selected.shape
                )
                heads[transfer_scope[selected]] = request_heads[
                    selected & transfer_scope
                ]
                # Reentering a return rebases the whole request, rather than
                # admitting its first nonnegative per-FFT clock offset. Keep
                # the leading skip separate from its decoded source head:
                # the latter is the successor allocation's physical start.
                # Changing delay can make a coarse request skip reject an
                # FFT that already passes the per-FFT lower bound.
                reentry = buffered_request_cursors(
                    _cursor_subtract(
                        self.desired_sample_starts[transfer],
                        state.anchor_samples[allocations[transfer]],
                        "successor request local origin",
                    ),
                    self.endpoint_drifts[transfer],
                    self.geometry,
                    sample_rate=self.sample_rate,
                )
                eligible[transfer] = (
                    active[transfer]
                    & reentry.valid[:, None]
                    & (positions[None, :] >= reentry.skipped_ffts[:, None])
                )
        physical_heads = _cursor_add(
            _cursor_multiply(
                timeline.allocation_starts[allocations],
                state.samples_per_frame,
                "retained physical allocation head",
            ),
            _cursor_multiply(
                cursors // self.geometry.samples_per_byte.denominator,
                self.geometry.samples_per_byte.numerator,
                "retained physical byte cursor",
            ),
            "retained physical window head",
        )
        contract = RetainedRawWindowContract(
            clock.first_fft,
            clock.output_samples,
            heads,
            clock.centre_delays_seconds,
            clock.framed_extent_samples,
            clock.decoded_extent_samples,
            np.broadcast_to(physical_heads[:, None], selected.shape)[
                selected
            ],
            eligible[selected],
        )
        if ordinary_head_contract is not None:
            if not isinstance(ordinary_head_contract, LeadingRawHeadContract):
                raise TypeError(
                    "ordinary head contract must be LeadingRawHeadContract"
                )
            if ordinary_head_contract.sample_rate != self.sample_rate:
                raise ValueError("ordinary head contract sample rate differs")
            if self.leading_head_inputs is None:
                raise ValueError(
                    "ordinary head contract requires endpoint clocks"
                )
            coordinates = compile_leading_raw_head_coordinates(
                self.leading_head_inputs,
                ordinary_head_contract,
                samples_per_frame=state.samples_per_frame,
            )
            ordinary = np.broadcast_to(
                (~events)[:, None] & (positions[None, :] == 0), selected.shape
            )[selected]
            ordinary &= (
                contract.aligned_head_samples == clock.aligned_head_samples
            )
            new_heads = np.broadcast_to(
                coordinates.head_sample_starts[:, None], selected.shape
            )[selected]
            physical = contract.physical_head_samples.copy()
            physical[ordinary] = _cursor_add(
                physical[ordinary],
                _cursor_subtract(
                    new_heads[ordinary],
                    contract.aligned_head_samples[ordinary],
                    "ordinary frame head clock shift",
                ),
                "ordinary physical frame head",
            )
            contract = replace(
                contract,
                aligned_head_samples=np.where(
                    ordinary, new_heads, contract.aligned_head_samples
                ),
                physical_head_samples=physical,
                decoder_clamps_negative_offsets=ordinary,
            )
        if not allocation_tail_ownership:
            return contract
        bounds = compile_raw_window_bounds(
            contract.output_samples,
            contract.aligned_head_samples,
            contract.centre_delays_seconds,
            contract.framed_extent_samples,
            sample_rate=int(self.sample_rate),
            fft_samples=self.geometry.fft_samples,
        )
        allocation = np.broadcast_to(
            batch.selected_allocations[:, None], selected.shape
        )[selected]
        event = np.broadcast_to(events[:, None], selected.shape)[selected]
        leading = np.broadcast_to(positions[None, :] == 0, selected.shape)[
            selected
        ]
        head_frames = timeline.allocation_starts[allocation]
        run = np.searchsorted(
            timeline.physical_stops, head_frames, side="right"
        )
        covered = run < timeline.physical_starts.size
        covered[covered] = (
            timeline.physical_starts[run[covered]] <= head_frames[covered]
        )
        physical = _cursor_add(
            contract.physical_head_samples,
            np.maximum(bounds.nearest_samples, 0),
            "retained tail physical start",
        )
        boundary = _cursor_multiply(
            _cursor_add(
                head_frames,
                state.allocation_frames,
                "retained allocation frame end",
            ),
            state.samples_per_frame,
            "retained allocation sample end",
        )
        unchanged_head = (
            contract.aligned_head_samples == clock.aligned_head_samples
        )
        withheld = (
            event
            & (leading | ~covered)
            & unchanged_head
            & (physical >= boundary)
            & ~bounds.missing
        )
        candidates = np.flatnonzero(
            event
            & unchanged_head
            & contract.publication_eligible
            & ~withheld
            & ~bounds.missing
        )
        # Framed admission includes header bytes at the lane's packing
        # ratio. Compare the actual decoder start, rather than only the
        # nominal per-FFT byte cursor. A retained predecessor cannot
        # publish beyond that extent across a forward anchor change.
        # Starts inside the header allowance remain admitted; payload
        # coverage and reported exposure are separate contracts.
        ratio = self.geometry.samples_per_byte
        framed_samples = -(
            -extent.allocation_framed_bytes
            * ratio.numerator
            // ratio.denominator
        )
        header_allowance = (
            framed_samples - state.allocation_frames * state.samples_per_frame
        )
        framed_boundary = _cursor_add(
            boundary[candidates],
            header_allowance,
            "retained framed allocation end",
        )
        candidates = candidates[physical[candidates] >= framed_boundary]
        selected_allocations = allocation[candidates]
        successor = selected_allocations + 1
        available = successor < state.anchor_samples.size
        candidates = candidates[available]
        expected = _cursor_add(
            state.anchor_samples[selected_allocations[available]],
            state.allocation_frames * state.samples_per_frame,
            "retained successor anchor",
        )
        withheld[candidates] = (
            state.anchor_samples[successor[available]] > expected
        )
        return apply_uncovered_head_publication(
            replace(
                contract,
                publication_eligible=(
                    contract.publication_eligible & ~withheld
                ),
            ),
            validity_state,
            extent=extent,
            sample_rate=int(self.sample_rate),
            fft_samples=self.geometry.fft_samples,
            publication_timestamp_quantum_nanoseconds=(
                None
                if ordinary_head_contract is None
                else ordinary_head_contract.timestamp_quantum_nanoseconds
            ),
        )


def evaluate_raw_job_requests(
    job: Job,
    datastream: Datastream,
    band: RecordedBand,
    frequency: Frequency,
    scan: Scan,
    geometry: BufferedRequestGeometry,
    *,
    first_fft: int,
    stop_fft: int,
    timeline_fft_count: int,
    job_start_slot: int,
    timestamp_origin_frame: int,
    maximum_requests: int = 4096,
    read_clock_convention: ReadClockConvention | None = None,
) -> RawJobRequestInputs:
    """Evaluate actual ordinary reads and the independent request contract.

    Ordinary reads use centre model selection, numpy nearest-even coarse
    rounding and actual output indices. Desired request starts use endpoint
    delay and add-half-then-truncate rounding. Caller must establish
    timestamp_origin_frame == job_start_slot; no raw/normalized conversion is
    invented here. Scope/request bounds are checked before any arrays or model
    evaluation, and arrays cover complete requests enclosing [first, stop).

    Opt-in endpoint-mean reads retain discrete-centre delays and output
    samples separately. Each endpoint selects its own active model; their
    arithmetic mean is neither a nonlinear integral nor a replacement for
    ordinary source coordinates, leading-head inputs, or fringe timing.
    """
    if not isinstance(geometry, BufferedRequestGeometry):
        raise TypeError("geometry must be BufferedRequestGeometry")
    if read_clock_convention is not None and (
        read_clock_convention is not ReadClockConvention.ENDPOINT_MEAN
    ):
        raise ValueError("unsupported read clock convention")
    first = _integer(first_fft, "first FFT")
    stop = _integer(stop_fft, "stop FFT", minimum=1)
    count = _integer(timeline_fft_count, "timeline FFT count", minimum=1)
    limit = _integer(maximum_requests, "maximum requests", minimum=1)
    for value in (job_start_slot, timestamp_origin_frame):
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError("timestamp origins must be signed integers")
    if timestamp_origin_frame != job_start_slot:
        raise ValueError("raw timestamp origin must equal job start slot")
    if not first < stop <= count:
        raise ValueError("selected scope must lie within the FFT timeline")
    width = geometry.ffts_per_request
    request_first = first // width * width
    request_count = (stop - request_first + width - 1) // width
    if request_count > limit:
        raise ValueError("request batch exceeds maximum requests")
    if count > np.iinfo(np.int64).max - width:
        raise OverflowError("FFT timeline exceeds int64 request arithmetic")
    rate = float(_sample_rate(datastream, frequency))
    if not np.isfinite(rate) or rate <= 0:
        raise ValueError("sample rate must be finite and positive")
    starts = request_first + np.arange(request_count, dtype=np.int64) * width
    counts = np.minimum(width, count - starts).astype(np.int64)
    origin = max(0.0, float(scan.start_offset_s))
    times = origin + starts.astype(np.float64) * (geometry.fft_samples / rate)
    indices = starts[:, None] + np.arange(width, dtype=np.int64)
    ordinary_times = origin + indices.astype(np.float64) * (
        geometry.fft_samples / rate
    )
    length, delays, coarse, output = _source_block_indices_for_scan(
        job, datastream, band, frequency, scan, ordinary_times.reshape(-1)
    )
    if length != geometry.fft_samples:
        raise ValueError("job FFT length differs from request geometry")
    if not np.all(np.isfinite(delays)):
        raise ValueError("ordinary model delays must be finite")
    ordinary = output - coarse
    if np.any(
        ((coarse > 0) & (ordinary > output))
        | ((coarse < 0) & (ordinary < output))
    ):
        raise OverflowError("ordinary source samples exceed int64")
    # Model-run lookup requires nondecreasing timestamps. Each endpoint
    # sequence is ordered; concatenating them would jump back to the start.
    centre_offset = (geometry.fft_samples - 1) / (2.0 * rate)
    _, left_delays, _, _ = _source_block_indices_for_scan(
        job, datastream, band, frequency, scan, times - centre_offset
    )
    _, right_delays, _, _ = _source_block_indices_for_scan(
        job,
        datastream,
        band,
        frequency,
        scan,
        times + width * geometry.fft_samples / rate - centre_offset,
    )
    if not (
        np.all(np.isfinite(left_delays)) and np.all(np.isfinite(right_delays))
    ):
        raise ValueError("request endpoint model delays must be finite")
    desired = np.asarray(
        [
            sample_offset_integer(time * rate)
            - sample_offset_integer(delay * rate)
            for time, delay in zip(times, left_delays, strict=True)
        ],
        dtype=np.int64,
    )
    drifts = right_delays - left_delays
    read_average = None
    if read_clock_convention is ReadClockConvention.ENDPOINT_MEAN:
        left = _endpoint_model_delays(
            job, datastream, band, scan, ordinary_times.reshape(-1)
        )
        right = _endpoint_model_delays(
            job,
            datastream,
            band,
            scan,
            (ordinary_times + geometry.fft_samples / rate).reshape(-1),
        )
        read_average = left * 0.5 + right * 0.5
        if not np.all(np.isfinite(read_average)):
            raise ValueError("read average model delays must be finite")
    return RawJobRequestInputs(
        first,
        stop,
        count,
        timestamp_origin_frame,
        geometry,
        starts,
        counts,
        desired,
        drifts,
        ordinary.reshape(request_count, width),
        rate,
        LeadingRawHeadInputs(
            output.reshape(request_count, width)[:, 0],
            left_delays,
            delays.reshape(request_count, width)[:, 0],
        ),
        read_clock_convention,
        delays.reshape(request_count, width)
        if read_clock_convention is not None
        else None,
        output.reshape(request_count, width)
        if read_clock_convention is not None
        else None,
        read_average.reshape(request_count, width)
        if read_average is not None
        else None,
    )


def bind_initial_retained_window_contract(
    inputs,
    validity_state,
    extent,
    *,
    normal_contract_factory,
):
    """Bind an explicitly declared initial return followed by normal returns.

    The caller selects this initial-publication contract separately from the
    normal-return policy. Pre-anchor requests refer to the first recorded
    allocation, with zero physical and logical heads. Whole-FFT skips use
    the actual request endpoint clock and drift. Per-FFT lower bounds,
    decoded coverage, and input validity remain separate execution checks.
    Wholly absent requests publish no FFT and never gain a valid cursor.

    This bounded contract requires an origin-zero contiguous first allocation;
    gaps, shifted origins and uncertain initial allocation ownership fail.
    It is not selected from a recording name, delay threshold or loss table.
    A separately supplied factory handles the enclosing normal-request suffix.
    No production execution profile is activated by this function.
    """
    from dataclasses import fields, replace

    from .clocks import LeadingRawHeadInputs
    from .geometry import BufferedNormalFullExtent
    from .replay import RetainedRawWindowContract
    from .validity import RawValidityState

    if not isinstance(inputs, RawJobRequestInputs):
        raise TypeError("initial binding requires RawJobRequestInputs")
    if not isinstance(validity_state, RawValidityState):
        raise TypeError("initial binding requires RawValidityState")
    if not isinstance(extent, BufferedNormalFullExtent):
        raise TypeError("initial binding requires explicit returned extent")
    if not callable(normal_contract_factory):
        raise TypeError("normal contract factory must be callable")
    source = validity_state.source_state
    state, timeline = source.request_state, source.timeline
    initial = inputs.desired_sample_starts < int(state.anchor_samples[0])
    if not np.any(initial):
        return normal_contract_factory(inputs)
    if np.any(np.diff(initial.astype(np.int8)) > 0):
        raise ValueError("initial requests must form one leading prefix")
    if inputs.read_clock_convention is None:
        raise ValueError("initial binding requires explicit model clocks")
    if state.timestamp_origin_frame != inputs.timestamp_origin_frame:
        raise ValueError("initial binding timestamp origins differ")
    if (
        int(state.anchor_samples[0]) != 0
        or int(timeline.allocation_starts[0]) != 0
    ):
        raise ValueError("initial source must start at the job clock origin")
    if (
        extent.samples_per_frame != state.samples_per_frame
        or extent.samples_per_byte != inputs.geometry.samples_per_byte
        or extent.request_frames * extent.read_buffer_ratio
        != state.allocation_frames
        or extent.framed_admission_samples.denominator != 1
    ):
        raise ValueError("initial extent and request geometry differ")
    cursor = 0
    for first, stop, timestamp in zip(
        timeline.physical_starts,
        timeline.physical_stops,
        timeline.timestamp_starts,
        strict=True,
    ):
        if int(first) != cursor or (
            int(timestamp) - state.timestamp_origin_frame != int(first)
        ):
            break
        cursor = int(stop)
        if cursor >= state.allocation_frames:
            break
    if cursor < state.allocation_frames:
        raise ValueError("initial allocation clock is not contiguous")

    width = inputs.geometry.ffts_per_request
    columns = np.arange(width, dtype=np.int64)
    indices = inputs.request_fft_starts[:, None] + columns
    selected = (
        (columns < inputs.request_fft_counts[:, None])
        & (indices >= inputs.first_fft)
        & (indices < inputs.stop_fft)
    )
    if not np.array_equal(
        indices[selected], np.arange(inputs.first_fft, inputs.stop_fft)
    ):
        raise ValueError("initial requests do not cover the selected scope")
    initial_scope = np.broadcast_to(initial[:, None], selected.shape)[
        selected
    ]
    initial_cursors = buffered_request_cursors(
        inputs.desired_sample_starts[initial],
        inputs.endpoint_drifts[initial],
        inputs.geometry,
        sample_rate=inputs.sample_rate,
    )
    eligible = np.zeros(selected.shape, dtype=np.bool_)
    eligible[initial] = (
        initial_cursors.valid[:, None]
        & (columns >= initial_cursors.skipped_ffts[:, None])
        & (columns < inputs.request_fft_counts[initial, None])
    )
    size = inputs.stop_fft - inputs.first_fft
    heads = np.zeros(size, dtype=np.int64)
    physical = np.zeros(size, dtype=np.int64)
    publication = eligible[selected]
    framed = np.full(
        size, int(extent.framed_admission_samples), dtype=np.int64
    )
    decoded = np.full(size, extent.decoded_payload_samples, dtype=np.int64)
    clamp = initial_scope.copy()
    if np.any(~initial_scope):
        # Preserve complete enclosing normal requests before scope clipping.
        rows = np.flatnonzero(~initial)
        updates = {}
        for field in fields(inputs):
            value = getattr(inputs, field.name)
            if isinstance(value, np.ndarray):
                updates[field.name] = value[rows]
        head = inputs.leading_head_inputs
        if head is not None:
            updates["leading_head_inputs"] = LeadingRawHeadInputs(
                head.request_output_samples[rows],
                head.endpoint_delays_seconds[rows],
                head.first_centre_delays_seconds[rows],
            )
        updates["first_fft"] = max(
            inputs.first_fft, int(inputs.request_fft_starts[rows[0]])
        )
        suffix = normal_contract_factory(replace(inputs, **updates))
        if not isinstance(suffix, RetainedRawWindowContract):
            raise TypeError(
                "normal factory must return retained window contract"
            )
        expected_first = inputs.first_fft + int(
            np.count_nonzero(initial_scope)
        )
        if (suffix.first_fft, suffix.stop_fft) != (
            expected_first,
            inputs.stop_fft,
        ):
            raise ValueError("normal contract must cover exactly the suffix")
        mask = ~initial_scope
        heads[mask] = suffix.aligned_head_samples
        physical[mask] = suffix.physical_head_samples
        publication[mask] = suffix.publication_eligible
        framed[mask] = suffix.framed_extent_samples
        decoded[mask] = suffix.decoded_extent_samples
        if suffix.decoder_clamps_negative_offsets is not None:
            clamp[mask] = suffix.decoder_clamps_negative_offsets
    return RetainedRawWindowContract(
        inputs.first_fft,
        inputs.output_sample_starts[selected],
        heads,
        inputs.centre_delays_seconds[selected],
        framed,
        decoded,
        physical,
        publication,
        clamp,
    )
