# Experimental native backend

The C++17/CUDA execution library lives under `src/cuphoton/xbeam/native`.
The standalone radio runner needs neither Python nor CuPy at runtime. The
HDF5 adapter uses Python for metadata and result serialization while keeping
raw reads and numerical execution in C++. Support is experimental. The optional
Cython interface exposes the experimental `native-correlate` command;
it is not a production `cuphoton xbeam correlate` backend.

Build the standalone runtime with a CUDA toolkit, OpenSSL development files,
and a C++17 compiler:

```bash
cmake -S src/cuphoton/xbeam/native -B build/xbeam-native \
  -DCMAKE_BUILD_TYPE=Release \
  -DXBEAM_ENABLE_CUDA=ON \
  -DXBEAM_ENABLE_CUFILE=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/xbeam-native -j
```

Python is used during the build to extract the existing cuPhoton-owned CUDA
kernel sources. Execution uses AOT kernels and cuFFT. With cuFile enabled, its
headers and library must be available in the CUDA toolkit. Omit that option
for a POSIX-only build.

Run an explicitly bounded integration window:

```bash
build/xbeam-native/xbeam-native-correlate JOB.input \
  --output NEW_OUTPUT_DIRECTORY \
  --index-dir SIDECARS \
  --scan SCAN_ID \
  --first-integration 0 \
  --integration-count 1 \
  --fft-tile 400 \
  --device 0 \
  --vdif-io cufile \
  --profile interop-real-usb-buffered-window-replay-v1
```

The runner loads `.input`, `.calc`, and `.im` metadata; validates existing
schema-v4 sidecars; computes buffered-window source selection, timing and
validity; reads physical frames; performs rotation, FFT, channelization and
XMAC; accumulates integrations; and writes a SWIN file plus `native-run.json`.
The output directory must be new. A failed run has no completion report.
A contiguous run writes `visibilities.swin`; dynamically scheduled runs write
`visibilities.chunk-NNNNNN.swin` files, listed in the completion report.
Choose `--vdif-io posix` for reads through pinned host memory. Both delivery
paths use the same GPU frame-payload extraction and correlation kernels.
cuFile sessions, file handles and device registrations persist across reads.
Direct reads and their owning GPU buffer views use 64-KiB alignment. Each
read chunk has its own persistent registration so delivery does not depend
on offsets into a larger registered region.
`--upload-workers` bounds concurrent reads for either I/O mode.
`--read-chunk-bytes` sets their maximum size (16 MiB by default), in
64-KiB multiples. More workers or larger reads can increase contention when
several GPUs share storage; tune these limits for the machine.
The standalone runner also supports `--read-staging shared` for cuFile.
It retains one registered raw staging buffer per upload worker and extracts
frame payloads before reusing that buffer, including payloads split across
read boundaries. The default `per-input` mode retains raw buffers with each
input slot. Both produce the same packed payload and validity buffers.
`--prefetch on` uses two input slots to overlap the next read with current
GPU work; `--prefetch off` provides a synchronous comparison. Both preserve
the same source selection and numerical contract.
XMAC packs products that share spectra, using their addresses and strides;
output positions and per-product normalization and weight rules are retained.
Failed or incomplete reads cannot publish an input buffer. Chunk tasks drain
before their input allocations are released.

Radio execution requires one real-sampled physical recording per datastream,
a shared FFT time grid, whole subintegrations per output integration, and
whole reader requests per tile. The profile remains explicit; unsupported
geometry fails rather than delegating to Python or substituting a profile.
Parallel-hand autocorrelations currently use the interoperability truncation
convention. Additional processing profiles, JSON-job loading, general sampling
geometry and shard publication remain open.
The optional CPU inspection binding builds with `CUPHOTON_XBEAM_BUILD_EXT=1`.
The default cuPhoton package build remains pure Python.

Inspection commands:

```bash
build/xbeam-native/xbeam-native inspect-vdif RECORDING.vdif
build/xbeam-native/xbeam-native inspect-job JOB.input
build/xbeam-native/xbeam-native inspect-index RECORDING.vdif INDEX.vdix
```

Component executables `xbeam-native-station-test` and
`xbeam-native-products-test` check CUDA execution. The bounded
`xbeam-native-station-file-test` diagnoses station frontend output; it does
not generate baseline visibilities. Component checks do not replace full-scan numerical and performance
qualification for a particular input and execution configuration.

To build the optional CUDA Cython interface, set
`CUPHOTON_XBEAM_BUILD_EXT=1 CUPHOTON_XBEAM_NATIVE_CUDA=1` during installation.
This binding links cuFFT and cuFile, so both development libraries are required.
`CUDA_HOME` selects the toolkit, and
`CUPHOTON_XBEAM_CUDA_ARCHITECTURES` can select explicit CMake CUDA targets for
builds without an accessible GPU. Then the same execution library is available
without launching a subprocess:

```python
from cuphoton.xbeam.native import correlate

report = correlate(
    "JOB.input", "NEW_OUTPUT_DIRECTORY", "SIDECARS", "SCAN_ID",
    first_integration=0, integration_count=1, fft_tile=1250,
    device=0, vdif_io="cufile",
    profile="interop-real-usb-buffered-window-replay-v1",
)
```

The CPU-only extension explicitly refuses correlation. It never silently
falls back to Python execution.
Python callers must supply `integration_count` and `profile` explicitly.
Input delivery defaults to buffered POSIX reads. Each call runs its selected
window to completion; use bounded windows for interactive applications.

With the CUDA extension installed, the command line calls the same Cython
entry point without entering Python radio loading, planning, or execution:

```bash
cuphoton xbeam native-correlate JOB.input \
  --output NEW_OUTPUT_DIRECTORY --index-dir SIDECARS \
  --scan SCAN_ID --first-integration 0 --integration-count 8 \
  --vdif-io cufile --fft-tile 1250 --upload-workers 2 \
  --profile interop-real-usb-buffered-window-replay-v1
```

Use the standalone executable when no Python runtime is desired. Native
outputs currently require separate shard assembly; they do not yet publish
the manifests required by `merge-shards`.

## CPU development checks

The CPU parser, reader planner, codec, and Cython inspection binding can be
checked without a GPU. Install the development extra, CMake, OpenSSL headers,
and a C++17 compiler, then run from a source checkout:

```bash
python -m pip install Cython setuptools-scm
cmake -S src/cuphoton/xbeam/native -B build/xbeam-native \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/xbeam-native -j
CUPHOTON_XBEAM_BUILD_EXT=1 python setup.py build_ext --build-lib build/native-binding
XBEAM_NATIVE_BINDING_DIR=build/native-binding python -m pytest tests/xbeam/native
```

Native binding tests are skipped when the optional binding is not supplied.
The CLI routing tests also run in a pure Python installation. Default package
builds do not require a CUDA toolkit or OpenSSL development files.

## Scheduling and I/O

For buffered POSIX, prefer contiguous integration windows per GPU to preserve
read locality. Budget CPU cores for all GPU workers on the node; limiting
four workers to one small CPU/NUMA subset can dominate input delivery time.
The standalone runner additionally offers `--vdif-io posix-direct`, which
uses aligned CPU `O_DIRECT` reads and pinned host-to-device copies. This is
separate from cuFile/GDS and is not automatically selected for POSIX.

Outputs should be written to node-local scratch before publication to shared
storage. Numerical acceptance should compare the complete selected integration
scope; output assembly and comparison are currently external to this backend.

The standalone runner can distribute one integration scope dynamically with
`--claim-dir DIRECTORY --workers-total N --min-chunk COUNT`. Create an empty
claim directory for that scope, give every worker the same job, scan,
integration range and scheduling options, and give each worker its own new
output directory. Workers claim chunks atomically and list their completed
files in `native-run.json`. Assembly must verify complete, nonoverlapping
coverage across all worker reports.

Claims remain assigned if a worker fails. Treat that campaign as incomplete
and use fresh claim and output directories when rerunning its scope. Claim
directories belong to one campaign and its common configuration.

## HDF5 time series

The native engine also computes generic time-series spectra and lag
correlations. HDF5 metadata is validated by the existing Python adapter;
C++ owns raw file reads, pinned/device buffers, cuFFT plans, cross-products,
validity statistics, and inverse transforms. Python serializes the owned
host results through the existing result writer. This path requires the
CUDA Cython extension and does not require CuPy at runtime.

Use the existing series configuration with `--backend native`:

```bash
cuphoton xbeam correlate-series series.json \
  --backend native --output results.h5 --upload-workers 2
```

```json
{
  "input": "signals.h5",
  "datasets": {"a": "/signals/a", "b": "/signals/b"},
  "io_backend": "posix",
  "correlation": {
    "pairs": [["a", "b"]],
    "window_samples": 4096,
    "hop_samples": 2048,
    "windows_per_result": 16,
    "batch_windows": 8,
    "output": "both",
    "lag_mode": "linear"
  }
}
```

Both `posix` and `cufile` use the same native numerical pipeline and the
existing time-series result schema. cuFile requires the existing aligned
HDF5 raw-layout contract. The native path currently requires contiguous,
allocated, native-endian float32/float64 or complex64/complex128 vectors;
optional Boolean validity vectors must also be contiguous. Compressed,
chunked, virtual, externally stored, and integer inputs are rejected rather
than delegated to a Python engine. Each signal retains its required
`sample_rate_hz` and `origin_seconds` attributes.

FP32/FP64 selection, overlapping or separated windows, pair ordering,
linear/circular lags, validity and exposure counts, and result sharding
follow the existing series API. Incomplete result groups are dropped.
The executor captures the CUDA device selected at construction; separate
processes can select GPUs with `CUDA_VISIBLE_DEVICES` and partition result
ordinals using `shard_count` and `shard_index` in `correlation`.

Reads use two reusable slots; `io_queue_depth` must currently be 2.
`read_buffer_bytes` bounds each raw input span, reducing the processing batch
when needed. A single window must fit that bound. `prefetch_batches=0`
disables the overlap. Results are host-owned; no borrowed device arrays
are exposed by this backend.
