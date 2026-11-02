# Time-series correlation

`correlate_series(source, spec)` accepts uniformly sampled real or complex
signals identified by strings. It yields cross-spectra, lag correlations,
or both. Use `ArraySource` for NumPy or CuPy vectors and `Hdf5Source` for
bounded reads from a file.

## Array API

```python
import numpy as np
from cuphoton.xbeam import ArraySource, CorrelationSpec, correlate_series

rng = np.random.default_rng(42)
a = rng.normal(size=8192).astype(np.float32)
b = np.zeros_like(a)
b[7:] = a[:-7]
source = ArraySource({"sensor-a": a, "sensor-b": b}, sample_rate_hz=1024)
spec = CorrelationSpec(
    pairs=(("sensor-a", "sensor-b"),),
    window_samples=1024,
    hop_samples=512,
    windows_per_result=4,
    output="both",
)
for result in correlate_series(source, spec):
    peak = result.lag_samples[np.argmax(abs(result.lag[0]))]
    print(result.result_index, peak)
```

`ArraySource` accepts NumPy or CuPy vectors. `backend="cupy"` executes
transforms and products on the GPU. Results default to host arrays;
`result_location="device"` keeps numerical products on the GPU.
`precision=None` preserves single or double floating input precision.
Mixed signals use the wider precision; integer inputs use double precision.
An explicit `precision="float32"` or `"float64"` overrides that choice.
GPU time-series and radio products use the same batched XMAC implementation
in `core/`. It fuses conjugation, multiplication, window reduction and
accumulation for single- and double-precision spectra. Normalization and
sample-validity accounting remain with their respective application paths.
Floating-point reduction order can differ from an unfused array expression;
bitwise identity is not a cross-backend guarantee.

## Numerical conventions

For each pair `(a, b)`, the cross-spectrum is
`FFT(a) * conj(FFT(b))`. Lag correlation is
`R_ab(k) = sum_n a[n] * conj(b[n+k])`: positive lag means that `b` arrives
later than `a`. Divide `result.lag_samples` by `result.sample_rate_hz` for
seconds. Peak selection returns a whole-sample delay.

`lag_mode="linear"` zero-pads to the next power of two at least `2*N-1`
and returns lags `-(N-1)` through `N-1`. All output selections use this FFT
grid. Frequencies follow standard FFT ordering, including negative
frequencies. `lag_mode="circular"` uses an N-point FFT and N centered lags,
so correlation wraps at the window boundary.

Products are raw sums averaged over the scheduled windows in each result.
Applications supply any taper, mean removal, calibration or normalization.
Spectrum-only requests skip the inverse transform and lag arrays. The radio
API has its own channelization, sideband and weighting conventions.

Optional Boolean validity vectors zero invalid samples before transforms.
Nonfinite samples marked valid are errors. Results report contributing
windows, simultaneous valid sample counts and, for lag output, valid
sample-pair counts at every lag. All-invalid pairs have `valid=False`.
Coverage counts are reported separately from products; a lag with zero
coverage has no measurement.

## HDF5 inputs and outputs

Each signal occupies a one-dimensional dataset with scalar `sample_rate_hz`
and `origin_seconds` attributes. Signals must share sample count, rate and
origin. The origin is an application-defined reference in seconds. The
adapter requires an aligned grid; applications perform any resampling or
clock correction before correlation.

The CLI accepts a JSON configuration:

```json
{
  "input": "signals.h5",
  "datasets": {"a": "/signals/a", "b": "/signals/b"},
  "validity": {"a": "/validity/a"},
  "correlation": {
    "pairs": [["a", "b"], ["a", "a"]],
    "window_samples": 1024,
    "hop_samples": 512,
    "windows_per_result": 4,
    "batch_windows": 16,
    "output": "both",
    "lag_mode": "linear"
  }
}
```

```bash
uv run cuphoton xbeam correlate-series series.json \
  --output results.h5 --backend numpy --upload-workers auto
```

Input paths are relative to the JSON file unless absolute. Dataset paths
and optional Boolean validity paths are explicitly configurable. Reads are
bounded to sample windows, including for compressed/chunked input.
Adjacent windows in a batch use
one HDF5 slice; overlapping windows share a bounded slice, and separated
windows do not read the gaps. Validation and GPU transfers operate on whole
batches. Unmasked inputs use exact analytical lag coverage instead of
allocating masks and transforming them. Explicit masks retain sample-level
coverage accounting.

`Hdf5Source(..., io_backend="posix")` is the default. With the HDF5 sec2
file driver, the source uses concurrent raw host reads when every signal and
validity dataset has an eligible contiguous, uncompressed native representation
in the opened file. Otherwise, the source uses HDF5 decoding for all datasets.
GPU execution uploads either form of host input. For native cuFile reads,
use `io_backend="cufile"` with `backend="cupy"`. HDF5 metadata is still read
through h5py; raw signal and validity extents are read into registered GPU
buffers with blocking, byte-count-checked `cuFileRead` calls on bounded host
I/O workers.
This requires the CUDA cuFile Python bindings,
a working GDS installation and a supported filesystem. Whether the storage
transport uses RDMA depends on the deployment.

The cuFile adapter requires the HDF5 sec2 file driver and allocated,
contiguous, uncompressed datasets
with native numeric types, no external/virtual storage, and dataset offsets
and byte sizes divisible by 4096. Boolean validity datasets have the same
alignment requirements. Create suitable files with, for example,
`h5py.File(path, "w", alignment_threshold=1, alignment_interval=4096)` and
choose vector lengths whose storage sizes are divisible by 4096. Unsupported
layouts raise an error; the adapter does not silently switch to POSIX.
Signal and validity datasets must belong to the opened file. Links to
datasets in another HDF5 file are rejected before CUDA resources are created;
same-file hard and soft links are accepted. POSIX also supports external links.
Before reading, cuFile checks that its opened file matches the file whose
HDF5 metadata was validated. A replaced source pathname raises an error.
The cuFile installation's own compatibility fallback is separately controlled
by its configuration; disable `allow_compat_mode` there to require direct I/O.

`io_queue_depth` defaults to 2 and accepts 2, 3, or 4 input slots.
Native POSIX and cuFile GPU input share window planning, ordered submission,
device slots, assembly, readiness checks and buffer reuse. Only the transport
differs: POSIX fills pinned host buffers and enqueues asynchronous CUDA copies;
cuFile reads into registered device buffers. Both use the same bounded range
scheduler across signals and queued batches and publish readiness only after
byte-count-checked I/O completes. All signals and validity arrays are submitted
together, with subsequent batches submitted before consumption.
`upload_workers` limits active CUDA I/O streams separately from these buffers.
For cuFile, it also bounds active read workers. Slots share streams when the worker limit is smaller than the queue depth.
`auto` retains one stream per slot; explicit stream limits are capped by queue depth.
For POSIX, the same setting separately limits the source-wide CPU read pool
(default four workers, up to sixteen), shared across signals and batches.
Ready events order compute after reads, and consumed events protect slot reuse. Completion is
checked on the host before a batch is consumed, so a failed or short read
raises instead of publishing partial data. The next slot is submitted ahead
of consumption, allowing its reads to overlap computation. Closing the source
drains all I/O streams before releasing registered buffers. `prefetch_batches=0`
uses serial cuFile reads with the same source-owned registered resources.
Queue-depth tuning applies to the asynchronous reader; disabling prefetch
also disables its deeper queue.

`read_buffer_bytes` defaults to 64 MiB **per dataset per GPU slot**, including
validity datasets. POSIX additionally allocates a pinned host buffer of that
size for each dataset in each GPU slot. Each batch's coalesced read envelopes must fit that limit.
Reduce `batch_windows`/`result_groups_per_tile` or increase the limit if needed.
FFT workspaces, gathered windows and output arrays require additional memory.
The JSON root accepts `io_backend`, `read_buffer_bytes`, `io_queue_depth`
and `upload_workers` with the same meaning. `upload_workers` accepts `auto`
or an integer from 1 through 16. The shared `--upload-workers` CLI option
overrides that JSON value, including an explicit `auto`. The GPU reader can
combine adjacent compute tiles into one storage block within that buffer
limit, then return each original tile without changing result grouping or
FFT sizes. Increasing the buffer limit can therefore enlarge reads without
increasing compute tiles. It never fills gaps between requested sample
windows. Mixed window sizes, changing signal selections and nonsequential
or irregular window requests end a block; metadata lookahead is also bounded. Both native POSIX
and cuFile use this block planner and shared buffer lifetime handling.
Larger queues multiply registered memory by the number of slots.

Each result is stored under `/results/<global-result-index>`, with pair
identifiers in `pairs_json`, sample bounds and grid metadata in attributes,
and arrays named after the `SeriesResult` fields. Optional products are
absent when not requested. Writing a device result explicitly copies its
arrays to host; GPU-resident output avoids those transfers only while the
consumer keeps processing on the GPU. Sample bounds cover all windows in the result;
`window_starts` identifies the actual windows when a hop leaves gaps.

## Streaming and independent machines

`result_groups_per_tile` defaults to 1. Increasing it batches several
independent result groups through the same FFT/product calls and output
transfer. It does not change `windows_per_result`, normalization, result
indices or shard boundaries. `batch_windows` limits windows **per group**
within a tile; memory therefore scales with both settings. Tail tiles can
contain fewer groups. Results own their output storage independently of input
slots, so retaining a result remains safe after advancing or closing the
iterator. Retaining one result can keep its entire output tile alive.

For downstream GPU processing:

```python
import cupy as cp
from cuphoton.xbeam import CorrelationSpec, Hdf5Source, correlate_series

spec = CorrelationSpec(
    pairs=(("a", "b"),), window_samples=65536,
    windows_per_result=16, batch_windows=16,
    result_groups_per_tile=8, backend="cupy", result_location="device",
    output="both",
)
consumer = cp.cuda.Stream(non_blocking=True)
with Hdf5Source("signals.h5", {"a": "/a", "b": "/b"},
                io_backend="cufile") as source:
    for result in correlate_series(source, spec):
        with consumer:
            result.wait(consumer)  # Required when using another CUDA stream.
            peak = cp.argmax(cp.abs(result.lag), axis=-1)
        # Transfer only the reduced peak indices and finish using this result.
        print(result.result_index, result.lag_samples[peak.get(stream=consumer)])
```

Spectrum, lag, validity, exposure and coverage arrays stay on the producing
GPU. Grid metadata (`window_starts`, `frequencies_hz`, `lag_samples`) remains
on the host. The iterator captures its producing device and stream when first
consumed; use `result.wait(stream)` to order a different consumer stream, and
keep results alive until that stream finishes using them. `to_host()` waits
for production, copies device arrays, and returns a host result. On a host
result it returns the same object. Device output requires the CuPy backend.

The default hop equals the window size and the default result contains one
window. Only complete windows and complete `windows_per_result` groups are
emitted. Incomplete trailing samples/groups are dropped. Batching bounds
working memory independently of total recording length. POSIX HDF5 sources read
one batch ahead by default (`prefetch_batches=1`). With the default queue depth of 2, the current and
next batch have independent host buffers. Increasing `io_queue_depth` allows
at most that many resident batches, including the current one and validity
arrays. For allocated contiguous datasets with standard native numeric or
Boolean storage, the POSIX path validates the raw extents once and reads
them in parallel outside HDF5's library lock. `upload_workers=auto` uses
four read workers; an explicit limit accepts 1 through 16. Each read is
split into chunks of at most 16 MiB, also bounded by `read_buffer_bytes`.
The workers fill independent buffer regions using positioned reads. A shared
bounded work queue issues contiguous chunks first, without waiting for an
entire signal or batch to finish before another can start.
A completed batch is published only after all its writers have drained. The source owns a duplicate of the open HDF5 file descriptor. On Linux, it
opens an independent file description per dataset through `/proc/self/fd`
to keep read-ahead state separate for distant datasets. It never reopens the
original filename. Where procfs is unavailable or restricted, duplicated
descriptors preserve correctness but share read-ahead state. All descriptors
remain owned until the read workers drain. Input files must remain
immutable while the source is open.

Compressed, chunked, virtual, externally linked, unallocated, non-native
or nonstandard encoded datasets retain HDF5 decoding. That fallback uses
one worker automatically; explicit limits are capped by queued batches
(`io_queue_depth - 1`), and HDF5 library locking may serialize file access.
The raw path requires the sec2 HDF5 driver and POSIX `preadv`; other
drivers retain HDF5 decoding. That decoded fallback uploads on the consuming
thread. Native POSIX reads instead fill pinned slots and enqueue uploads on
the same I/O streams used by the common device pipeline, allowing reads,
copies and computation to overlap. Readiness and buffer-reuse events remain
necessary to prevent incomplete inputs and overwritten buffers. Set `prefetch_batches=0` for synchronous reads. This is a scheduling option, not another I/O backend.
Read-ahead stays within the selected shard and complete result groups.
Closing the result iterator or source cancels queued work and waits for an
in-flight read before closing the HDF5 file. Read errors propagate when the
affected batch is consumed. This does not interrupt blocked filesystem I/O. The caller owns
source lifetime and should consume the iterator inside the source context.

For `correlate-series`, `shard_count` and `shard_index` are keys in the JSON
`correlation` object. Set `shard_count` to the number of tasks. Each task needs
its own configuration, differing only in `shard_index`, and its own output
file. Indices run from zero to `shard_count - 1`; reusing one configuration
would repeat that shard instead of producing the other results. All tasks
read the same input. Select a local GPU through `CUDA_VISIBLE_DEVICES` and
`--backend cupy`.

`plan_series_shards(sample_count, spec)` partitions **global result ordinals**,
so overlapping windows read their required halo but are emitted exactly once.
Empty shards are valid. Combine HDF5 result groups by their global ordinal;
never average shard outputs together. There is no distributed HDF5 writer or
resampling/clock-calibration facility in this interface.

## Implementation boundaries

`core/` owns format-independent contracts, array products, and fused GPU
cross-product kernels. `runtime/` owns backend loading, execution policy, and
the shared CUDA module cache. `adapters/arrays.py` and `adapters/hdf5.py` provide
series inputs; `adapters/radio/` and `adapters/vdif/` own application orchestration
and recording-specific interpretation. Radio spectral products use shared
numerical primitives;
there is no second copy of its correlation engine.

Adapters may optionally implement
`read_windows(signal_id, starts, window_samples)` to return a two-dimensional
sample batch and an optional Boolean batch of the same shape. Returning no
mask means all samples in that batch are valid; nonfinite valid samples
still fail validation. The core otherwise falls back to bounded `read`
calls. Adapters must keep returned arrays valid through batch consumption.
An adapter can declare `required_backend="cupy"` when its input buffers
require GPU execution; incompatible configurations fail before reading.

Host-backed adapters can opt into the same bounded scheduler by providing
`prefetch_batches(requests)`, a context manager yielding ordered batch
mappings. Each request contains `(signal_ids, starts, window_samples)`.
Each yielded mapping contains the `(samples, optional_mask)` pair for every
requested signal. The adapter must return independently owned host buffers
and keep its resources alive until the worker joins. GPU-backed adapters
must provide their own stream/event ownership before opting in; no GPU
work is submitted by the generic host prefetch worker. A device batch mapping
can expose `wait()` and `release()`: the core waits before accessing arrays
and releases in a `finally` block after submitting consumption on its compute
stream. The adapter must validate read completion and delay reuse until that
stream's consumed event completes. Sources without prefetch capability keep
their synchronous behavior.

An adapter with both host and device input can additionally provide
`prefetch_device_batches(requests)`. GPU execution selects that capability;
CPU execution continues to use `prefetch_batches`. HDF5 implements both using
the same extent planner and ordered queue, with a shared device-slot lifecycle
for native POSIX and cuFile input. Adapters provide byte ranges and assemble
completed buffers into samples. VDIF ingest uses the same positioned-read,
buffer-allocation, copy and completion primitives while retaining its frame
decoding, validity rules and sample-frontier scheduling.
