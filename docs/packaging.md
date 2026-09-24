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

## Versions and release candidates

The build derives the package version from Git tags using `setuptools-scm`:

| Tag | Installed version |
| --- | --- |
| `v0.1.3rc0` | `0.1.3rc0` |
| `v0.1.3rc1` | `0.1.3rc1` |
| `v0.1.3` | `0.1.3` |

Use exactly `vX.Y.Z` or `vX.Y.ZrcN`. A checkout after a release tag receives
a development version. There is no version constant to bump: builds generate
`cuphoton/_version.py`, and package metadata, `cuphoton.__version__`, and
`cuphoton --version` use that version. A source archive preserves it without
Git. An unbuilt source checkout reports `0.0.0.dev0` until installed or built;
a Git-free copy without archive metadata cannot produce a release.

Release CI explicitly selects the triggering tag's version. This also works
when an RC tag and a final tag refer to the same commit. A final release needs
a new build and qualification because its version and metadata change; do not
rename RC wheels.

An exact RC pin works without `--pre`:

```bash
python -m pip install 'cuphoton[dev,gpu,io,photometry]==0.1.3rc0'
```

`gpu` already includes `io` and `photometry`, so `[dev,gpu]` is equivalent.
Add `viz` for visualization dependencies. Use `--pre` when selecting the newest
available prerelease instead of pinning one. Each changed candidate needs a
new RC number: PyPI does not allow replacing an uploaded filename.

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

Configure these Trusted Publishers in the respective package-index accounts:

| Setting | PyPI | TestPyPI |
| --- | --- | --- |
| Project | `cuPhoton` | `cuPhoton` |
| Repository owner | `NVIDIA` | `NVIDIA` |
| Repository | `cuPhoton` | `cuPhoton` |
| Workflow filename | `publish.yml` | `publish.yml` |
| GitHub environment | `pypi` | `testpypi` |

Require a reviewer for each GitHub environment and allow deployments from
branch `main` and tags matching `v*`. Keep self-review available when the release
operator is the sole reviewer. Project owners register each publisher on its
index; GitHub environment configuration alone does not grant upload access.
No stored API token is needed. See the [PyPI Trusted Publisher setup
instructions](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

Pushing `v0.1.3rc0` or `v0.1.3` starts `publish.yml`. It validates the tag and
requires its commit to belong to `main` or `0.1.x`, builds the six native wheels
from one versioned source archive, and tests their clean installation. It then
waits at the `pypi` environment for approval. No release tags are created by the
workflow.

1. Download `cuphoton-distributions` and `cuphoton-build-provenance` from the
   release run. The latter records the tag, source and workflow commits, build
   run/attempt, and every distribution's SHA256. Artifacts expire after 30 days.
2. Qualify those exact binaries on both GPU architectures as described above.
   The environment reviewer checks those results and hashes before approving
   the upload. CPU CI success does not establish GPU correctness.
3. To rehearse on TestPyPI before approving PyPI, dispatch `publish.yml` from
   `main` with the same `version`, `target=testpypi`, and the original release
   `run-id`. This reuses its artifacts even while its PyPI job awaits approval.
4. Verify the TestPyPI downloads against the recorded hashes, then approve the
   original PyPI job. Alternatively, cancel that waiting job before dispatching
   with `target=pypi` and the same original build run ID. Publishing runs for
   one version and destination are serialized, so leaving the original waiting
   would block the replacement. The replacement promotes the identical files.

Manual dispatch without `run-id` builds the supplied existing tag and publishes
to the selected environment after approval. Dispatch with `run-id` always uses
the original **build run**, including when its upload failed or was cancelled;
a promotion-only run has no build artifacts of its own. To rebuild, start a
new dispatch; uploaded artifacts are immutable, so rerunning an already
completed build in the same run cannot replace them. The source tag must
still resolve to the recorded commit, and version, artifact hashes, matrix,
and the build attempt's successful validation must all match.

A retry first compares files already on the selected index with the retained
artifacts. Matching files are omitted from the upload; a differing hash or
unexpected filename fails. Publication never rebuilds artifacts or silently
accepts different bytes. The upload job alone receives the OIDC permission.
