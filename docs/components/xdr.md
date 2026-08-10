# xDataReader

`cuphoton.xdr` provides GPU-oriented FITS and HDF5 loading. Its FITS
path uses native CFITSIO planning plus KvikIO, nvCOMP, and CuPy to load
supported image HDUs directly to device arrays. Its optional HDF5 path
delegates to Legate's integrated HDF5 API and returns a Legate
`LogicalArray`.

Current scope:

- uncompressed image HDUs
- `GZIP_1` and `GZIP_2` compressed image HDUs
- batched multi-file loading with `batch_to_device`
- pipelined loading with `batch_to_device_stream`
- one-dataset HDF5 loading with `load_hdf5`
- explicit `NotImplementedError` for compression formats that do not have a GPU
  path

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

Normal installation attempts to build the native extension and falls back to
the pure Python package when native prerequisites are unavailable. Set
`CUPHOTON_XDR_BUILD_EXT=1` to require the extension or
`CUPHOTON_XDR_BUILD_EXT=0` to skip it explicitly. Only CUDA 13
dependency variants are supported.

### Native extension availability

Wheels built from this repository are pure Python (`py3-none-any`) and never
contain `cuphoton.xdr._nvcomp_batch_ext`; the extension is built only from a
source checkout. The build intentionally runs without PEP 517 build isolation
(as `build.sh` does with `--no-build-isolation`) because pybind11 and the
KvikIO, nvCOMP, and CFITSIO headers and libraries are resolved from the
installed `gpu` environment — this is also why `pybind11` is not listed in
`[build-system].requires`. Verify the extension after building:

```bash
uv run python -c "from cuphoton.xdr.nvcomp_batch import cpp_helper_available; print(cpp_helper_available())"
```

Without the extension, `batch_to_device` and `batch_to_device_stream` raise a
`RuntimeError` that names the missing module and the build command.

Install the optional Legate HDF5 backend alongside the development profile:

```bash
uv sync --locked --extra dev --extra hdf5
```

## HDF5 loading with Legate

`load_hdf5` is a thin integration with
[`legate.io.hdf5.from_file`](https://docs.nvidia.com/legate/latest/api/python/generated/legate.io.hdf5.from_file.html):

```python
from cuphoton.xdr import load_hdf5

images = load_hdf5("observation.h5", "/images/science")
```

The returned object is a Legate `LogicalArray`. Execution is asynchronous;
use Legate's runtime fence when the caller needs an explicit completion
boundary. Resource placement, distributed partitioning, GDS, and virtual
dataset behavior belong to the installed Legate runtime. For example, a
GDS-enabled Legate build can be configured before launching Python:

```bash
export LEGATE_CONFIG="--gpus 1 --io-use-vfd-gds"
```

xDataReader imports Legate only when this API is called, so the base package
and the FITS loader remain usable without the `hdf5` extra. Custom Legate
builds with experimental parallel or virtual-dataset readers can be installed
into the same environment without changing the xDataReader API.

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
