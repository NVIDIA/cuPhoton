# Native wheels

Release artifacts are six Linux wheels (CPython 3.12, 3.13, and 3.14 on
x86-64 and ARM64) plus one source archive. Wheels target glibc 2.28 or later.
The runtime dependencies may impose a newer glibc floor; the installed-wheel
CI tests use Debian 12. Free-threaded Python, Windows, macOS, and conda builds
are outside this matrix.

Each wheel contains `cuphoton.xdr._nvcomp_batch_ext` and a privately renamed,
reentrant CFITSIO 4.7.0 shared library. CUDA, cuFile, KvikIO, and nvCOMP remain
in their upstream wheels, installed through `cuphoton[io]`. `cuphoton[gpu]`
also includes the photometry, PyTorch, and Numba backends. Photutils currently
requires a source build and C compiler on ARM64; `cuphoton` and `cuphoton[io]`
do not install it.

The native KvikIO ABI is restricted to the 26.6 release family. nvCOMP is
restricted to 5.2. Updating either family requires rebuilding and qualifying
the native wheels. The CUDA SDK build inputs are pinned to 13.0 so a newer
build environment cannot silently raise the runtime floor.

## Build

From the checkout, on each native Linux architecture with Docker and uv:

```bash
make wheels
python scripts/wheels/check_distributions.py dist --arch "$(uname -m)"
uvx --from twine==6.2.0 twine check --strict dist/*
```

`make wheels` builds a source archive, then uses cibuildwheel 4.2.1 to build
all three Python ABIs from that archive. It uses digest-pinned manylinux
images, the build inputs in `scripts/wheels/build-requirements.txt`, and the
checksum-pinned CFITSIO recipe in `scripts/wheels/prepare_cfitsio.sh`.
CFITSIO's upstream tests run before installation. No host CUDA toolkit is
used. Release builds must not set `CIBW_TEST_SKIP`.

The default `make build` produces only the source archive. Plain source and
editable installs remain Python-only unless `CUPHOTON_XDR_BUILD_EXT=1` is
set. See [XDR source installation](components/xdr.md#native-extension-availability)
for the explicit native development build.

The reusable `wheels.yml` workflow builds from one source archive on native
x86-64 and ARM64 runners. It checks base imports inside cibuildwheel, then
installs each wheel and its `io` dependencies in a clean Python container
with no compiler, system CFITSIO, or CUDA toolkit. Native loading and CFITSIO
planning must work without a GPU. The final artifact check requires exactly
six native wheels and one source archive; it rejects accidental pure wheels,
missing native code or notices, and bundled GPU runtime libraries.

## GPU qualification

CI CPU checks do not establish GPU correctness. Download the exact
`cuphoton-distributions` artifact and test each wheel on its architecture and
Python version with a CUDA 13-compatible driver. Use a clean runtime container
without a compiler, system CFITSIO, or a local toolkit. Install the wheel
with its `io` extra, then run from outside any checkout:

```bash
python -m pip install '/artifacts/cuphoton-<version>-<tags>.whl[io]'
python -m pip check
python -I /checks/test_installed.py --mode gpu --output /results/gpu.json
```

`test_installed.py` lives under `scripts/wheels` in the source archive. The
GPU check requires native loading and decoding, compares generated FITS
images with Astropy, and exercises ordering, ROI, streaming, caller-owned
outputs, concurrency, and buffer release. Missing GPU/native capabilities
fail the check. It forces KvikIO compatibility I/O, so it does not qualify
GPUDirect Storage. Qualify GDS separately on suitable host/storage systems.

Retain the source commit, artifact SHA256 values, container image, installed
dependency versions, Python, GPU/driver details, and JSON test receipts. When
testing the minimum CUDA runtime, constrain `cuda-toolkit==13.0.3.0` and
`nvidia-nvjitlink==13.0.88`; also test the normal unconstrained `io` resolution.

## Publish the qualified artifacts

Configure PyPI and TestPyPI Trusted Publishers for this repository,
`publish.yml`, and the respective `pypi` / `testpypi` GitHub environments.
Require a reviewer for each environment and restrict publishing to `main`.
No stored API token is needed. See the [PyPA Trusted Publishing workflow
guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/).

1. Select a successful **push** run of `ci.yml` on `main` or `0.1.x` and
   download its `cuphoton-distributions` artifact before its 30-day expiry.
2. Qualify those exact binaries on both GPU architectures. Retain their
   hashes and results for the environment reviewer.
3. Create the approved `v<version>` tag at that run's source commit.
4. Dispatch `publish.yml` from `main`, supplying that run ID, version, and
   `testpypi`. The workflow verifies the CI provenance, tag, complete matrix,
   and metadata, and displays the artifact hashes before environment approval.
5. Verify TestPyPI downloads against those hashes and repeat the dispatch
   with `pypi` after release approval.

Publication downloads and uploads the verified CI artifacts without rebuilding
them. A successful CI run does not replace GPU qualification or release
approval. The publishing workflow itself does not create tags or releases.
