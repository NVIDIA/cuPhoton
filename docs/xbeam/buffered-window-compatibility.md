# Buffered-reader replay

Select buffered-reader source selection and FFT admission explicitly:

```console
cuphoton xbeam correlate JOB --output OUTPUT --backend cupy \
  --profile interop-real-usb-buffered-window-replay-v1 --vdif-io posix
```

Use `--vdif-io cufile` for GPUDirect Storage. The numerical policy is the same
on CPU and GPU. Ordinary `interop-real-usb-v1` correlation does not apply this
replay policy.

## Required inputs

- Real-sampled VDIF, including interleaved channels.
- Exactly one physical recording and one physical thread per datastream.
- Declared reader buffer settings and consistent sample geometry.
- Valid schema-v4 sidecars containing physical timestamp runs.
- Known initial and returned-window source history.

A logical thread alias uses the physical mapping validated by ingest.
Multiplexed physical threads, malformed headers, inconsistent formats, and
unsupported or unknown history fail explicitly. Rebuild older sidecars before
selecting replay; see [sidecars](sidecars.md).

## Numerical behavior

Plans derive from reader geometry, validated physical timestamp runs, and job
delay and clock models. Recording names, station labels, scan identifiers,
and measured output corrections do not select behavior.

Physical sample addresses, logical timing, FFT admission, and sample validity
are separate quantities. Storage translation does not change phase, and
admission does not make invalid payload valid. The retained allocation clock
stays latched while physical payload is consumed; later header timestamps do
not automatically replace it.

If frames are missing within a retained allocation, this policy continues to
consume the following physical payload under the latched clock until the next
allocation anchor. Affected visibilities can therefore be assigned earlier
logical times and retain full reported weight. The reported weight follows
the replay admission policy; it does not establish that every timestamp slot
was recorded. Ordinary timestamp-grid correlation instead marks missing
slots invalid and aligns subsequent payload using its recorded timestamps.

A validated physical file end bounds decoded payload, including trailing
filler, independently of replay tile size. Samples beyond that end are
unavailable. Negative or otherwise unknown retained history remains an error.
Publication uses the declared integer-nanosecond clock separately from the
precise decoder clock; that quantization does not change voltage addresses or
residual-delay phase.

Replay does not recover absent samples. It applies buffered-reader admission
and source-selection semantics, which can differ from ordinary timestamp-grid
correlation. It is not a universal recorder policy or a guarantee of agreement
with every external correlator configuration.
