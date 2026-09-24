# xDataReader

`cuphoton.xdr` provides GPU-oriented FITS loading. It uses native CFITSIO
planning plus KvikIO, nvCOMP, and CuPy to load supported image HDUs directly
to device arrays.

Current scope:

- uncompressed image HDUs
- `GZIP_1` and `GZIP_2` compressed image HDUs
- batched multi-file loading with `batch_to_device`
- pipelined loading with `batch_to_device_stream`
- explicit `NotImplementedError` for unsupported compression formats

## HDF5 migration

`cuphoton.xdr.load_hdf5` and the `hdf5` installation extra have been removed.
For general HDF5 access, use `h5py` directly; it is a base dependency. Remove
`hdf5` from installation extras, for example by replacing `cuphoton[hdf5]`
with `cuphoton`.

For supported XRay detector inputs, see the [single-node HDF5
workflow](../xray/README.md#single-node-hdf5-workflow), which uses the `h5py`
reader by default.

## Install

Install the CUDA 13 development profile:

```bash
uv sync --locked --extra dev --extra gpu
```

The base package supports imports in CPU environments. xDataReader's loading
paths require the `gpu` extra. To build its native extension after syncing
the GPU environment, run:

```bash
bash src/cuphoton/xdr/src/build.sh
```

The native extension also needs CUDA toolkit headers, cuFile headers, and a
thread-safe CFITSIO development install visible through `pkg-config cfitsio` or
`CUPHOTON_XDR_CFITSIO_ROOT`.

Normal PEP 517 and pip builds produce the pure Python package. Set
`CUPHOTON_XDR_BUILD_EXT=1` for a native source build, as `build.sh` does above.
`CUPHOTON_XDR_BUILD_EXT=0` explicitly selects the default pure Python build.
The supported dependency variants target CUDA 13.

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
support. Keep `--enable-reentrant` for concurrent native planning and reads.

Official release wheels are pure Python (`py3-none-any`). Build
`cuphoton.xdr._nvcomp_batch_ext` from source using the installed `gpu`
environment: `build.sh` passes `--no-build-isolation` to resolve pybind11,
KvikIO, and nvCOMP from that environment. CFITSIO headers and libraries come from
`CUPHOTON_XDR_CFITSIO_ROOT` or `pkg-config cfitsio`, as described above.
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

`benchmark-fits --mock-storage {device,host}` serves repeat reads of each
file from an in-memory cache, so runs measure decode and kernel cost
independent of disk throughput. `device` replays from GPU memory at HBM
bandwidth, isolating decompression cost and modeling an ideally fast GDS
path; `host` replays from pinned host memory over PCIe, modeling what a
properly working GDS path would deliver on the same hardware. The same
behavior is available programmatically through the
`cuphoton.xdr.mock_storage` context manager, or transparently by setting
`CUPHOTON_XDR_MOCK_STORAGE=device` or `host` to set the benchmark's default.
