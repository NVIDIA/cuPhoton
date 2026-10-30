# Data contract

Normalized jobs use schema version 1 and are represented by immutable model
objects in `cuphoton.xbeam.adapters.radio.models`. JSON jobs must match the
[job schema](../../src/cuphoton/xbeam/schemas/job-v1.schema.json); unknown fields and malformed scalar values are
rejected.

A job declares:

- UTC-like time as MJD plus decimal seconds of day;
- stations and their clock polynomials;
- frequencies, sidebands, channel counts, and averaging;
- datastream frame geometry, recorded bands, and optional zoom bands;
- baseline products and scan delay-coordinate models;
- integration and subintegration durations; and
- native or SWIN output format.

`Scan.station_models` may contain multiple delay/UVW polynomial epochs for
each station. All stations used by a scan must declare the same epoch grid.
The model at the greatest epoch not later than an FFT block centre supplies
that block's delay, while the model active at an integration midpoint supplies
its UVW coordinates. A one-epoch scan is the backward-compatible special
case. The first epoch must cover the part of the scan inside the job execution
interval.

The execution engine yields `VisibilityRecord` objects. Spectra are complex
arrays accompanied by time, baseline, frequency, polarization, weight, and
UVW metadata. Writers validate their format-specific fields before encoding.

Input adapters implement the `RecordingSource` protocol and yield
`VoltageBatch` objects on the job sample grid. A batch may contain decoded
samples or frame-packed two-bit payloads described by `PackedVoltageLayout`.
Multiple batches for one band must have a common representation and may not
overlap. File-backed VDIF adapters validate frame, thread, station, channel,
sample-rate, and bounded timeline geometry before allocating payload buffers.
Caller-provided batches may cover only part of the requested interval; missing
boundary and delay-margin samples are zero-filled and reduce sample validity.
Built-in ingest instead reserves those margins and checks allocation coverage
as an internal invariant.
