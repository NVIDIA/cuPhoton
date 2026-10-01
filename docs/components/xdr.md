# xDR

`cuphoton.xdr` provides GPU-oriented FITS loading. It uses native CFITSIO
planning plus KvikIO, nvCOMP, and CuPy to load supported image HDUs directly
to device arrays.

Current scope:

- uncompressed image HDUs
- `GZIP_1` and `GZIP_2` compressed image HDUs
- batched multi-file loading with `batch_to_device`
- pipelined loading with `batch_to_device_stream`
- explicit `NotImplementedError` for unsupported compression formats

## Runtime choices

Three independent controls select how xDR restores supported compressed FITS
pixels. Each defaults to `auto`. They preserve image values, mask bits and
floating-point bit patterns; choose explicit modes for comparisons or debugging.

| Direct Python keyword / `xdr_options` key | CLI flag | Values | Meaning of `auto` |
| --- | --- | --- | --- |
| `postprocess` | `--xdr-postprocess` | `auto`, `fused`, `separate` | Fuse unshuffle, byte-order conversion and scatter. |
| `gzip_decoder` | `--xdr-gzip-decoder` | `auto`, `gzip`, `deflate` | Use native Gzip when available; otherwise decode aligned raw DEFLATE. |
| `decompression_backend` | `--xdr-decompression-backend` | `auto`, `cuda` | Let nvCOMP choose a compatible decompression engine for native Gzip, with CUDA fallback. |

`postprocess="fused"` selects the same kernel as `auto`; `"separate"` runs
individual restoration kernels. `gzip_decoder="gzip"` requires native Gzip
support. `"deflate"` strips the Gzip framing and aligns payloads for raw
DEFLATE. `decompression_backend="cuda"` requires a native extension with
backend selection support and selects CUDA kernels explicitly. The native
raw DEFLATE decoder uses CUDA in either backend mode.

Both postprocessing modes leave the decoded input buffer unchanged. The
separate path allocates another buffer the size of the decoded tiles and
copies GZIP_1 input before byte-order conversion. Its timings therefore
include that copy and are not an exact baseline for the former in-place
GZIP_1 implementation.

### Direct Python APIs

Both batch APIs and `GpuCompImageReader.read` accept these named keywords:

```python
from cuphoton.xdr import batch_to_device

(images,) = batch_to_device(
    ["exposure-1.fits", "exposure-2.fits"],
    hdu_indices=[1],
    postprocess="auto",
    gzip_decoder="auto",
    decompression_backend="auto",
)
```

The same controls apply with `batch_to_device_stream` or `parallel=False`.
Python prefetching (`native_batcher=False`, or CLI `--native-batcher off`)
still uses the selected GPU decoder. It does not select CPU decompression.

### Workflow APIs and manifests

Workflow APIs accept the three keys in an `xdr_options` mapping. For example,
to require xDR and compare CUDA decompression with separate pixel restoration:

```python
from cuphoton.core.fits_io import read_fits_images

result = read_fits_images(
    "exposure.fits", [1], reader="xdr", device=True,
    xdr_options={
        "postprocess": "separate",
        "gzip_decoder": "gzip",
        "decompression_backend": "cuda",
    },
)
image = result.arrays[0]
```

The mapping is also accepted by the FITS-consuming APIs in
[xFit](xfit.md), [xPois](xpois.md), [xRep](xrep.md), and [xScan](xscan.md).
Manifest-based workflows preserve their input choices through worker
serialization and execution. See the component guide for the mapping's
placement in its manifest or FITS descriptor.

An explicit API override or CLI flag replaces only the corresponding manifest
key. Omitted flags preserve manifest values; keys absent from both use `auto`.
Passing `--xdr-gzip-decoder auto` explicitly therefore replaces a manifest's
`gzip_decoder` value. Reader receipts include explicit `xdr_options`; they
record requested settings rather than which nvCOMP engine actually ran.

### Reader selection and fallback

`--fits-reader` (Python `reader` or `fits_reader`) selects Astropy versus xDR.
The three controls above apply after that selection and do not request a GPU
reader by themselves. See [reader policies below](#read-fits-images-in-a-workflow)
for automatic CPU fallback and FITS semantic restrictions.

Within xDR, native Gzip with `decompression_backend="auto"` may use CUDA when
a hardware engine or suitable buffers are unavailable. An older extension
without native Gzip support can use aligned raw DEFLATE. The low-level decoder's
Python fallback also runs on the GPU and retains nvCOMP's existing backend
defaults. Full batch FITS loading
still requires the native FITS planner; codec fallback does not remove that
requirement.

When xDR decodes compressed HDUs, explicit `gzip` or `cuda` requests fail clearly
if the installed extension cannot honor them. Rebuild an older source extension
using the [native build instructions](#native-extension-availability). The `auto`
choices select available capabilities before decoding. I/O or decoder exceptions
propagate; they do not trigger another reader, codec or postprocessing attempt.
The supported FITS compression formats remain `GZIP_1` and `GZIP_2`.

### Hardware eligibility and diagnostics

nvCOMP selects an engine for each native Gzip call. Its
[Decompression Engine FAQ](https://docs.nvidia.com/cuda/nvcomp/decompression_engine_faq.html)
lists B200, B300, GB200 and GB300 support. GB10 and RTX PRO 6000 Blackwell
use CUDA kernels; the Blackwell name alone does not imply engine support.
The compressed data, output and decoded-size buffers must all
use compatible allocations. B200 has a 4 MiB hardware chunk limit; the limit
on a device is available through `CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH`.
Use `decompression_backend="cuda"` when comparing execution paths or reading
tiles beyond the hardware limit.

Batch readers retain native pooled scratch buffers. The low-level non-pooled
path, including `GpuCompImageReader.read()` without an explicit stream, uses
`cudaMallocAsync` scratch, which is not hardware-decompression capable. That
path selects CUDA directly, including with `decompression_backend="auto"`,
to avoid a failed hardware attempt on each call. Custom CuPy allocators can
also affect eligibility in pooled calls.
An `auto` receipt and a compatible GPU therefore do not establish hardware use.
See [nvCOMP logging](../troubleshooting.md#confirm-the-decompression-engine)
to inspect the actual decoder calls.

The benchmark's `--output-json` report includes
`capabilities.hardware_decompression`: the current device's algorithm mask,
`supports_deflate`, and `max_chunk_bytes`. Unavailable queries leave those
values `null` and record an `error`. These device limits are collected after
the timed phases and do not identify the engine used by an individual call.

The native decoder expects valid compressed payloads and correct output sizes.
Header validation and a successful launch do not verify payload integrity;
nvCOMP's [C API](https://docs.nvidia.com/cuda/nvcomp/c_api.html)
does not guarantee safe decoding of corrupt Gzip or DEFLATE streams.

### Tile size and batch size

Gzip tiles provide independent work for the decoder. A batch with only a few
large tiles can leave much of the GPU idle, including on GPUs without a
hardware decompression engine. Increasing `decode_batch_files` can supply
more tiles per decode call when GPU memory allows it.

For newly written FITS files, start with the usual row-sized tiles or tiles
of a few tens of KiB, then measure the complete read with representative
data. In one 128 MiB workload, 8–64 KiB tiles gave similar read times;
multi-MiB tiles were much slower. That result is a starting point, not a
universal optimum. Tiles beyond the device's hardware chunk limit use CUDA
in automatic mode; changing the backend alone does not restore the missing
tile parallelism.

## Read FITS images in a workflow

The shared FITS reader selects explicit image HDUs and returns NumPy or CuPy
arrays. Inspecting the headers does not decompress image pixels:

```python
from cuphoton.core.fits_io import inspect_fits_images, read_fits_images

planes = inspect_fits_images("exposure.fits")
result = read_fits_images(
    "exposure.fits", [planes[0].hdu], reader="auto", device=True
)
image = result.arrays[0]  # CuPy array, ready for use
print(result.metadata())  # Actual reader and any fallback reason
```

The Python reader defaults to `reader="astropy"` and `device=False`.
`reader="astropy"` decodes on the CPU; `reader="xdr"` requires the GPU reader.
`reader="auto"` uses xDR for supported lossless images when its dependencies,
native planner, and CUDA device are available. Scaling, integer nulls,
quantization, unsupported compression, and externally compressed files use
Astropy. Read errors propagate after the selected reader starts. With
`device=False`, the returned arrays reside on the host, including an explicit
download when xDR decoded them. Integer masks retain their width and bits.

Use `section=(slice(y0, y1), slice(x0, x1))` for bounded cutouts with unit-step
slices. Automatic reads of uncompressed cutouts use Astropy's section access;
the shared reader rejects explicit xDR requests for those sections.
Supported tile-compressed cutouts can use xDR. Device output requires CuPy
and a usable CUDA device even when Astropy performs decoding. Device reads
finish before returning, including reads on an explicitly supplied CuPy stream.

The pipeline and applicable standalone FITS workflows expose this reader
policy. Prepared NPY, NPZ, and HDF5 inputs retain their existing readers.
Reader receipts describe decoded arrays and reader selection; they do not
measure physical storage traffic. Establish native GDS with process-local
cuFile counters for the measured reads, distinguishing P2PDMA/NVFS from POSIX
fallback. The legacy `is_gds_active` probe requires `nvidia-fs` and can report
false on working P2PDMA configurations that do not use that module.

## Load a batch onto the GPU

The lower-level xDR APIs load the same HDU indices from every input file.
Files must have matching shapes and dtypes at each selected HDU:

```python
from cuphoton.xdr import batch_to_device

(images,) = batch_to_device(
    ["exposure-1.fits", "exposure-2.fits"], hdu_indices=[1]
)
first_image = images[0]
```

The result is a tuple of CuPy arrays, one per selected HDU, each with shape
`(file, y, x)`. `batch_to_device_stream` exposes the same stacked result with
prefetch, decode-batch and queue controls. Both accept preallocated `out`
arrays and a CuPy `stream`. Their `section` option applies to compressed
image HDUs. Use the shared reader above when you need automatic Astropy
fallback and FITS semantic checks. The exported `open_gpu` function is not
implemented; use one of these readers instead.

## HDF5 migration

`cuphoton.xdr.load_hdf5` and the `hdf5` installation extra have been removed.
For general HDF5 access, use `h5py` directly; it is a base dependency. Remove
`hdf5` from installation extras, for example by replacing `cuphoton[hdf5]`
with `cuphoton`.

For supported xRay detector inputs, see the [single-node HDF5
workflow](../xray/README.md#single-node-hdf5-workflow), which uses the `h5py`
reader by default.

## Install

Version 0.1.3 is a source-only release. Start with a
[source checkout](../getting-started.md#clone-and-select-a-profile) and build
the native extension as shown below. The `io` extra installs CuPy, KvikIO,
cuFile, and nvCOMP; `gpu` also includes these dependencies.
The base package remains importable without GPU dependencies.
Only CUDA 13 dependency variants are supported. A compatible NVIDIA driver
is required for GPU execution.

GPUDirect Storage also needs a supported host driver, filesystem, and storage
configuration. KvikIO compatibility mode supports ordinary local file I/O.
Use `KVIKIO_COMPAT_MODE=ON` to select compatibility mode explicitly.

From the checkout:

```bash
uv sync --locked --extra dev --extra io
bash src/cuphoton/xdr/src/build.sh
```

Source and editable builds default to a Python-only installation without
probing native prerequisites. Building the extension requires
`CUPHOTON_XDR_BUILD_EXT=1`, as performed by `build.sh` above. The source build
requires a C++17 compiler, CUDA and cuFile headers, and a reentrant CFITSIO
development installation. See [native extension availability](#native-extension-availability)
below for prerequisites and [Packaging](../packaging.md) for local package builds.

### Native extension availability

`build.sh` requires `uv` on `PATH` and prints the interpreter it selects.
Selection order is `PYTHON`, the active `VIRTUAL_ENV`, the checkout's
`.venv/bin/python`, then `python3` or `python` on `PATH`. An invalid explicit
interpreter or active environment fails before installation. To select the
interpreter running a command, use:

```bash
PYTHON="$(uv run python -c 'import sys; print(sys.executable)')" \
  bash src/cuphoton/xdr/src/build.sh
```

For a bare CUDA 13 development container, install a C/C++ compiler, `make`,
`pkg-config`, zlib development headers, and the CUDA/cuFile development headers.
If the system CFITSIO package lacks thread support, build a reentrant copy
from the [CFITSIO source distribution](https://heasarc.gsfc.nasa.gov/docs/software/fitsio/):

```bash
# Run in a writable build directory outside the checkout.
export CUPHOTON_XDR_CFITSIO_ROOT="$PWD/cfitsio-install"
curl -fLO https://heasarc.gsfc.nasa.gov/FTP/software/fitsio/c/cfitsio-4.7.0.tar.gz
tar -xzf cfitsio-4.7.0.tar.gz
(
  cd cfitsio-4.7.0
  ./configure --prefix="$CUPHOTON_XDR_CFITSIO_ROOT" \
    --enable-reentrant --disable-curl
  make -j4
  make check
  make install
)
```

Return to the checkout and run `build.sh` in the same shell so the prefix
remains exported. The prefix must contain `include/fitsio.h` and the CFITSIO
library in `lib` or `lib64`. `--disable-curl` removes CFITSIO's optional URL
support; local FITS loading does not require it. Keep `--enable-reentrant`
for concurrent native planning and reads.

An explicit native source build runs without
PEP 517 build isolation (as `build.sh` does with `--no-build-isolation`)
because pybind11, KvikIO and nvCOMP are resolved from the installed `io`
environment. CFITSIO headers and libraries come from
`CUPHOTON_XDR_CFITSIO_ROOT` or `pkg-config cfitsio`, as described above.
The helper installs pybind11 as a build dependency; it is not an I/O runtime
requirement.
Verify the extension after building:

```bash
uv run python -c "from cuphoton.xdr.nvcomp_batch import cpp_helper_available; print(cpp_helper_available())"
```

If the extension is missing, `batch_to_device` and `batch_to_device_stream`
raise a `RuntimeError` that names the missing module and the build command.

## Benchmark

```bash
cuphoton xdr benchmark-fits \
  --hdu-indices 1,2,3 --output-json benchmark.json \
  /path/to/file1.fits /path/to/file2.fits
```

All three [runtime controls](#runtime-choices) are available on this command.
For a comparison against separate restoration and the native CUDA DEFLATE path,
repeat the same workload with a distinct report:

```bash
cuphoton xdr benchmark-fits \
  --hdu-indices 1,2,3 --output-json separate-deflate-cuda.json \
  --xdr-postprocess separate --xdr-gzip-decoder deflate \
  --xdr-decompression-backend cuda \
  /path/to/file1.fits /path/to/file2.fits
```

Use the same files, HDUs, batching settings and storage mode for both runs.
These controls affect the `batch_to_device` and `batch_to_device_stream`
phases; planner and raw-read timings do not measure decompression. To isolate
one change, vary only that control. The Python `run_benchmark` API accepts
`postprocess`, `gzip_decoder` and `decompression_backend` directly.

Use `--dir` and `--max-files` to scan directories of FITS files. The benchmark
defaults to `--native-read-threads=4`; the loading APIs default to the available
CPU core count when `native_read_threads` is omitted.

`--output-json` writes an optional report while retaining the terminal table.
Its parent directory must exist. The report replaces its destination only after
all phases finish; preflight or uncaught errors leave a previous report intact.
A caught phase failure still produces a report with `ok=false` and the phase's
error text. The Python API accepts `output_json=Path(...)` and continues
returning `list[PhaseResult]`.

The version 1 JSON report contains:

- `phases`: the same full-precision measurements returned by the benchmark,
  including per-phase `ok` and `error` fields. `elapsed_ms` is the mean over
  that phase's iterations.
- `versions`: cuPhoton, Python, CuPy, KvikIO, and nvCOMP package versions.
- `capabilities`: native-helper import availability and the existing
  `is_gds_active` capability probe. `gds_active=true` does not prove that every
  measured read used GDS, including when storage is mocked.
- `storage`: the effective `real`, `host`, or `device` mode, including an
  ambient mock-storage context or environment setting.
- `options`: HDUs, iteration count, thread/queue settings, native-batcher
  selection, and requested `postprocess`, `gzip_decoder` and
  `decompression_backend`. An `auto` value does not identify which nvCOMP
  engine ran or prove hardware decompression. `native_batcher_enabled=null`
  records an invalid forced selection
  with the reason in `native_batcher_error`.
- `workload`: ordered input files, counts, planned raw bytes, and decoded MiB.
  Failed planning can leave these planned sizes at zero.

Metadata and JSON writes sit outside phase timings. `--skip-gds-read` omits the
raw-read phase; failed planning also prevents that phase from running.

### Benchmarking with cached input

`benchmark-fits --mock-storage {device,host}` preloads an in-memory cache
before its timed phases. `device` replays from GPU memory to measure decode
and kernel costs
without disk reads; `host` replays from pinned host memory and includes
host-to-device transfer. Neither mode measures native GDS or storage
throughput. The same behavior is available programmatically through the
`cuphoton.xdr.mock_storage` context manager, or transparently by setting
`CUPHOTON_XDR_MOCK_STORAGE=device` or `host` to set the benchmark's default.
In a custom benchmark, populate the cache before measuring repeat reads;
the first access otherwise includes a real file read.
