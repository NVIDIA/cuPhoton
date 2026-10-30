# xBeam

`cuphoton.xbeam` computes cross-spectra and lag correlations for uniformly
sampled signals, and radio-interferometry visibilities from VDIF recordings.
Both workflows support NumPy on CPU and CuPy on CUDA 13. The CLI group is
`cuphoton xbeam`.

## Install and try it

Run a synthetic HDF5 example from the cuPhoton checkout:

```bash
uv sync --locked --extra dev
uv run python examples/xbeam/correlate_series.py /tmp/series-example
```

The example generates white noise and a copy delayed by seven samples,
writes the inputs and results to HDF5, and reports the measured delay.
Use a new output directory for each run. For GPU execution:

```bash
uv sync --locked --extra dev --extra io
uv run python examples/xbeam/correlate_series.py /tmp/series-gpu \
  --backend cupy
```

See [time-series correlation](time-series.md) for array inputs, HDF5 layout,
lag conventions, validity masks and independent result shards.

## Correlate radio recordings

A radio job describes station clocks, spectral bands, baseline products,
scans and their delay models. Pass a normalized JSON job or supported DiFX
`.input` metadata. Correlation reads each recording through a header index
(a sidecar). Build a sidecar for every recording first with
`build-vdif-sidecar`, or pass `--assume-ordered` for recordings you have
independently checked are in time order:

```bash
uv run cuphoton xbeam correlate JOB --output OUTPUT --backend numpy
uv run cuphoton xbeam correlate JOB --output OUTPUT --backend cupy \
  --vdif-io posix
```

The output directory must be new. Native `xbeam-v1` output and SWIN output
carry visibility spectra, timestamps, weights and baseline metadata. See
the [data contract](data-contract.md) and
[numerical profiles](conformance.md) for their interpretation.

To build header indexes in a shared directory and use them for correlation:

```bash
uv run cuphoton xbeam build-vdif-sidecar RECORDING.vdif \
  --index-dir SIDECARS --backend posix
uv run cuphoton xbeam correlate JOB --output OUTPUT --backend cupy \
  --index-dir SIDECARS --scan SCAN --first-integration 0 \
  --integration-count 100 --ingest-window-integrations 10 \
  --max-read-bytes 1073741824
```

`--max-read-bytes` limits each ingest window. CuPy tiles oversized
integrations within device memory; `--max-tile-subintegrations` caps that
tile size. Use `--vdif-io cufile` with a working cuFile installation and
compatible storage. The [sidecar guide](sidecars.md) covers source validation
and index lookup; [buffered-reader replay](buffered-window-compatibility.md)
describes the optional reader policy and its additional input requirements.

## Use multiple GPUs

Add `--devices 0,1` to a CuPy radio correlation command to divide its scan
selection into contiguous time shards. Device numbers refer to the inherited
`CUDA_VISIBLE_DEVICES` allocation. Each worker enforces its own read limit.
The coordinator validates and merges the completed parts; individual parts
remain under `OUTPUT/shards/`.

An external scheduler can launch tasks with `--shard-count` and
`--shard-index`, then assemble their outputs with:

```bash
uv run cuphoton xbeam merge-shards JOB /shared/run/shard-* \
  --output /shared/run/complete
```

All shards must have compatible execution contracts and valid integrity
metadata. Use `cuphoton xbeam help COMMAND` for the available options.

Run the component's synthetic tests with `make test-xbeam`. The
[visibility benchmark](../../examples/xbeam/benchmark_visibility.py)
compares the GPU reduction against a deterministic NumPy calculation and
reports steady-state timings on caller-selected synthetic shapes.

## Benchmarks

Run the synthetic visibility benchmark from the repository root:

```bash
python -m examples.xbeam.benchmark_visibility --help
```

The GDS examples read files supplied by the caller. Use
`python -m examples.xbeam.benchmark_gds --help` to measure transfers and
`python -m examples.xbeam.benchmark_gds_ingest --help` to exercise the
correlation input pipeline without writing visibilities. Both require the
`io` extra and a working CUDA and GPUDirect Storage configuration.
