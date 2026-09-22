# xDataReader

`cuphoton.xdr` provides GPU-oriented FITS loading. It uses native CFITSIO
planning plus KvikIO, nvCOMP, and CuPy to load supported image HDUs directly
to device arrays.

Current scope:

- uncompressed image HDUs
- `GZIP_1` and `GZIP_2` compressed image HDUs
- batched multi-file loading with `batch_to_device`
- pipelined loading with `batch_to_device_stream`
- explicit `NotImplementedError` for compression formats that do not have a GPU
  path

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

The base package remains importable without GPU dependencies. xDataReader's
loading paths require the `gpu` extra. To require a local build of its native
extension after syncing the GPU environment, run:

```bash
bash src/cuphoton/xdr/src/build.sh
```

The native extension also needs CUDA toolkit headers, cuFile headers, and a
thread-safe CFITSIO development install visible through `pkg-config cfitsio` or
`CUPHOTON_XDR_CFITSIO_ROOT`.

Normal PEP 517 and pip builds produce the pure Python package without probing
native prerequisites. Building the extension requires an explicit
`CUPHOTON_XDR_BUILD_EXT=1` source build, as performed by `build.sh` above.
`CUPHOTON_XDR_BUILD_EXT=0` explicitly selects the default pure Python build.
Only CUDA 13 dependency variants are supported.

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

Official release wheels are pure Python (`py3-none-any`) and do not contain
`cuphoton.xdr._nvcomp_batch_ext`. An explicit native source build runs without
PEP 517 build isolation (as `build.sh` does with `--no-build-isolation`)
because pybind11 and the KvikIO, nvCOMP, and CFITSIO headers and libraries are
resolved from the installed `gpu` environment — this is also why `pybind11` is
not listed in `[build-system].requires`. Verify the extension after building:

```bash
uv run python -c "from cuphoton.xdr.nvcomp_batch import cpp_helper_available; print(cpp_helper_available())"
```

Without the extension, `batch_to_device` and `batch_to_device_stream` raise a
`RuntimeError` that names the missing module and the build command.

## Benchmark

```bash
cuphoton xdr benchmark-fits \
  --hdu-indices 1,2,3 /path/to/file1.fits /path/to/file2.fits
```

Use `--dir` and `--max-files` to scan directories of FITS files.

### Benchmarking without storage I/O

`benchmark-fits --mock-storage {device,host}` serves repeat reads of each
file from an in-memory cache instead of storage, so runs measure decode and
kernel cost independent of disk throughput. `device` replays from GPU memory
at HBM bandwidth, isolating decompression cost and modeling an ideally fast
GDS path; `host` replays from pinned host memory over PCIe, modeling what a
properly working GDS path would deliver on the same hardware. The same
behavior is available programmatically through the
`cuphoton.xdr.mock_storage` context manager, or transparently by setting
`CUPHOTON_XDR_MOCK_STORAGE=device` or `host` for benchmarks that do not
select it explicitly.
