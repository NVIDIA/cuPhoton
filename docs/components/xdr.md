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

The default `postprocess="auto"` uses fused unshuffle, byte-order conversion,
and scatter for supported GZIP FITS tiles. `"fused"` selects that path explicitly;
`"separate"` runs those steps with individual kernels for comparison.
Both preserve FITS values, including integer masks and floating-point bits.
Both leave the decoded input buffer unchanged. The separate path allocates
another buffer the size of the decoded tiles and copies GZIP_1 input before
byte-order conversion. Its timings therefore include that copy and are not
an exact baseline for the former in-place GZIP_1 implementation.

```python
from cuphoton.xdr import batch_to_device

(images,) = batch_to_device(paths, postprocess="auto")
```

Workflow FITS APIs accept `xdr_options={"postprocess": "separate"}` alongside
`fits_reader="xdr"` (or `reader="xdr"` for `read_fits_images`). FITS input
descriptors can persist the same `xdr_options` mapping. The corresponding
CLI flag is `--xdr-postprocess {auto,fused,separate}`, including on
`cuphoton xdr benchmark-fits`. Explicit CLI values override matching descriptor
fields; omitted flags preserve the descriptor's choices.

These controls apply when the existing reader policy selects xDR. Reader
selection and CPU fallback remain controlled by `--fits-reader`. Invalid options
fail during validation; decode, I/O, and CUDA failures propagate to the caller.

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

After the first PyPI release is published, install the I/O profile for GPU
FITS loading. Until then, use the source checkout instructions below:

```bash
python -m pip install 'cuphoton[io]'
```

Linux x86-64 and ARM64 wheels for CPython 3.12–3.14 include the native
extension and a private, reentrant CFITSIO 4.7.0 library. The extra installs
CuPy, KvikIO, cuFile, and nvCOMP; `gpu` also includes these dependencies.
The base package remains importable without GPU dependencies. No compiler,
local CUDA toolkit, or system CFITSIO is needed for a wheel installation.
Only CUDA 13 dependency variants are supported. A compatible NVIDIA driver
is required for GPU execution.

GPUDirect Storage also needs a supported host driver, filesystem, and storage
configuration. KvikIO compatibility mode supports ordinary local file I/O;
installing a wheel does not configure GDS. Use `KVIKIO_COMPAT_MODE=ON` to
select compatibility mode explicitly.

For development from a source checkout:

```bash
uv sync --locked --extra dev --extra io
bash src/cuphoton/xdr/src/build.sh
```

Source and editable builds default to a Python-only installation without
probing native prerequisites. Building the extension requires
`CUPHOTON_XDR_BUILD_EXT=1`, as performed by `build.sh` above. The source build
requires a C++17 compiler, CUDA and cuFile headers, and a reentrant CFITSIO
development installation. See [the wheel build procedure](../packaging.md)
for the pinned release recipe.

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
- `options`: HDUs, iteration count, thread/queue settings, and native-batcher
  selection. `native_batcher_enabled=null` records an invalid forced selection
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
