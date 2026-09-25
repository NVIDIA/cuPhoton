# Native packages

Release artifacts are six Linux wheels (CPython 3.12, 3.13, and 3.14 on
x86-64 and ARM64) plus one source archive. Wheels target glibc 2.28 or later.
The runtime dependencies may impose a newer glibc floor; the installed-wheel
CI tests use Debian 12. Native conda builds cover the same six Python and
architecture combinations, as described below. Free-threaded Python, Windows,
and macOS are outside both matrices.

Each wheel contains `cuphoton.xdr._nvcomp_batch_ext` and a privately renamed,
reentrant CFITSIO 4.7.0 shared library. CUDA, cuFile, KvikIO, and nvCOMP remain
in their upstream wheels, installed through `cuphoton[io]`. `cuphoton[gpu]`
also includes the photometry, PyTorch, and Numba backends. Installing Photutils
from PyPI currently requires a source build and C compiler on ARM64;
`cuphoton` and `cuphoton[io]` do not install it. Conda provides ARM64 Photutils
binaries.

The native KvikIO ABI is restricted to the 26.6 release family. nvCOMP is
restricted to 5.2. Updating either family requires rebuilding and qualifying
the native packages. The CUDA SDK build inputs are pinned to 13.0 so a newer
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

## Build wheels

From the checkout, on each native Linux architecture with Docker and uv:

```bash
make wheels
python scripts/wheels/check_distributions.py dist --arch "$(uname -m)"
uvx --from twine==6.2.0 twine check --strict dist/*.tar.gz dist/*.whl
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

`make build` refreshes only the source archive. `make wheels` replaces cuPhoton
wheels while preserving conda outputs; `make conda` replaces `dist/conda` while
preserving wheels. Use `make clean-dist` explicitly to remove all distributions.
To reuse the wheel build's exact archive for conda, use the direct conda build
command below.

The reusable `wheels.yml` workflow builds from one source archive on native
x86-64 and ARM64 runners. It checks base imports inside cibuildwheel, then
installs each wheel and its `io` dependencies in a clean Python container
with no compiler, system CFITSIO, or CUDA toolkit. Native loading and CFITSIO
planning must work without a GPU. The final artifact check requires exactly
six native wheels and one source archive; it rejects accidental pure wheels,
missing native code or notices, and bundled GPU runtime libraries.

The installed-wheel runtime matrix also requires cuTile imports and two real
MPI workers on every Python/architecture pair. Python 3.12 and 3.13 require
two real Dragon workers as well; upstream Dragon has no Python 3.14 wheel.
These workers solve generated xPOIS inputs on the CPU and check numerical
results, distinct processes, and MPI collectives. JSON receipts are retained
as CI artifacts. These checks do not establish GPU executor correctness.

## Build and install conda packages

The `cuphoton` conda package includes the compiled XDR extension and depends on
upstream conda packages for CFITSIO 4.7, CUDA 13, KvikIO 26.6, and nvCOMP 5.2.
Conda installs those libraries into the environment; no manual CUDA paths
are needed.
Conda has no pip-style extras: this package provides the native I/O profile
only and requires CUDA 13 libraries. It does not provide a CPU-only install;
use the base pip package for that profile.

With uv and pixi installed, run on each native Linux architecture:

```bash
make conda
```

This builds the SCM-versioned source archive and uses rattler-build 0.76.1 to
produce Python 3.12, 3.13, and 3.14 packages. The archive supplies the same
version used by wheels, including RC versions, without requiring Git during
the conda build. To build one Python version from an existing source archive:

```bash
pixi exec --spec rattler-build=0.76.1 --spec python=3.12 -- \
  python scripts/conda/build.py 'dist/cuphoton-<version>.tar.gz' \
  --python 3.12 --output-dir dist/conda
```

Use an output directory without existing cuPhoton packages. Outputs go under
`linux-64` or `linux-aarch64`, with a `provenance.json` recording the source
archive and package SHA256 values. Builds use strict channel priority with
`rapidsai` before `conda-forge`, excluding `defaults`. This order is required
by the current installed-package solver; it also gives RAPIDS precedence for
other packages present in both channels. Review the resolved environment
before qualifying an artifact.
The qualified conda nvCOMP build is 5.2.0.10, while the wheel environment uses
5.2.0.13. Both stay within the required 5.2 ABI family and need independent
artifact qualification; their patch versions are not interchangeable evidence.

Create a local channel index so conda resolves the artifact's runtime
dependencies. For downloaded CI artifacts, place the packages under
`dist/conda/linux-64` or `dist/conda/linux-aarch64` first. Select the version
and build string from the package filename:

```bash
mkdir -p dist/conda/noarch
pixi exec --spec conda-index=0.13.0 -- python -m conda_index dist/conda
conda create --prefix ./conda-xdr --override-channels \
  --strict-channel-priority -c "$PWD/dist/conda" -c rapidsai -c conda-forge \
  'python=3.12' 'cuphoton=<version>=<build>'
```

Direct installation of a `.conda` filename skips dependency resolution; use
the indexed channel directory above. See [conda's installation
guidance](https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/installing-with-conda.html).
For broader GPU and photometry functionality, add the following packages.
The PyTorch build selector chooses its CUDA 13.0 build:

```bash
conda install --prefix ./conda-xdr --override-channels \
  --strict-channel-priority -c "$PWD/dist/conda" -c rapidsai -c conda-forge \
  'photutils>=3' 'numba>=0.61,<0.66' 'numba-cuda>=0.30,<0.31' \
  'pytorch>=2.13,<3' 'pytorch=*=cuda130*' 'cuda-version=13.0'
```

Install development tools such as `pytest`, `ruff`, and `pre-commit` through
conda. Keep the repository's uv development environment separate: the current
conda PyTorch package requires `setuptools<82`, while the pip `dev` extra
requires `setuptools>=83`. For Python-only editable work in a separate conda
environment with the runtime dependencies and pip installed, run
`CUPHOTON_XDR_BUILD_EXT=0 conda run --prefix ./conda-dev python -m pip install --no-deps -e .`.
Native builds should use the recipe above.

The reusable `conda.yml` workflow builds all six variants from one source
archive and retains the packages and provenance as CI artifacts. Its isolated
installation tests load the native extension and exercise CFITSIO planning
without a GPU or driver. Upstream conda GPU packages retain some pip-only
dependency names in their Python metadata, so `pip check` is not a conda
validation gate. The actual native and GPU tests are required.

Qualify each exact conda artifact on its architecture and Python version with
a CUDA 13-compatible driver, using the installed-package GPU check:

```bash
conda run --prefix ./conda-xdr python -I scripts/wheels/test_installed.py \
  --mode gpu --output conda-gpu.json
```

The check runs outside the source tree internally and exercises the same
decoding, ordering, concurrency, and buffer-lifetime cases as wheels. Retain
its JSON receipt and the tested artifact hash. Conda channel publication is
not configured; release-tag publishing below applies to PyPI artifacts.

## Wheel GPU qualification

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

## Whole-stack qualification

Use `scripts/wheels/test_stack.py` alongside the native GPU check above.
Install the same wheel with `[gpu,viz,cutile,mpi,dragon]` on Python 3.12 and
3.13, or `[gpu,viz,cutile,mpi]` on Python 3.14. An MPI runtime and matching
launcher are required. cuTile uses an external CUDA 13.2 or newer compiler;
see [optional runtime setup](getting-started.md#optional-runtimes) for the
current PyTorch/compiler dependency constraint.

From outside the checkout, using the installed environment's Python:

```bash
python -I /checks/test_installed.py --mode gpu --output /results/xdr.json
python -I /checks/test_stack.py --mode gpu --report /results/compute.json

# One rank per visible GPU; this example needs two CUDA 13-capable GPUs.
CUDA_VISIBLE_DEVICES=0,1 timeout --kill-after=15s 180s mpiexec -n 2 \
  cuphoton-openmpi-rank-exec -- python -I /checks/test_stack.py \
  --mode mpi --backend cupy --workers 2 --report /results/mpi-gpu.json

# Python 3.12 or 3.13; two available GPU placements are required.
CUDA_VISIBLE_DEVICES=0,1 timeout --kill-after=15s 180s dragon --single-node-override \
  python -I /checks/test_stack.py --mode dragon --backend cupy \
  --workers 2 --report /results/dragon-gpu.json
```

For a one-GPU host, use one visible device, `mpiexec -n 1`, and `--workers 1`
for both executors. Record that as single-worker acceptance, not multi-GPU
qualification. Use `--backend cpu --workers 2` with the two launchers to
repeat the CPU runtime checks without GPU requirements.

The compute check solves the same known xPOIS problem with CPU, CuPy,
Numba-CUDA, and cuTile, and checks CuPy-to-PyTorch GPU inference through
xScan's DLPack bridge. The executor checks run cuPhoton's real MPI/Dragon
batch paths and verify saved numerical outputs. Missing selected runtimes,
GPU support, worker results, or compiler tools fail instead of skipping.
Retain these JSON receipts with the wheel hashes and native XDR receipts.

## Publish the qualified artifacts

Configure these Trusted Publishers in the respective package-index accounts:

| Setting | PyPI | TestPyPI |
| --- | --- | --- |
| Project | `cuPhoton` | `cuPhoton` |
| Repository owner | `NVIDIA` | `NVIDIA` |
| Repository | `cuPhoton` | `cuPhoton` |
| Workflow filename | `publish.yml` | `publish.yml` |
| GitHub environment | `pypi` | `testpypi` |

Require at least one reviewer other than the release operator for each GitHub
environment and enable **Prevent self-review**. Allow deployments from branch
`main` and tags matching `v*`. Protect release tags with a ruleset that restricts
creation to authorized release maintainers and prevents tag updates and deletion.
Project owners register each publisher on its index; GitHub environment
configuration alone does not grant upload access.
No stored API token is needed. See the [PyPI Trusted Publisher setup
instructions](https://docs.pypi.org/trusted-publishers/adding-a-publisher/).

Pushing `v0.1.3rc0` or `v0.1.3` starts `publish.yml`. It validates the tag and
requires its commit to belong to `main` or `0.1.x`, builds the six native wheels
from one versioned source archive, and tests their clean installation. It then
waits at the `testpypi` environment for approval. PyPI publication requires an
explicit manual dispatch with `target=pypi`. No release tags are created by the
workflow.

1. Download `cuphoton-distributions` and `cuphoton-build-provenance` from the
   release run. The latter records the tag, source and workflow commits, build
   run/attempt, and every distribution's SHA256. Artifacts expire after 30 days.
2. Qualify those exact binaries on both GPU architectures as described above.
   The environment reviewer checks those results and hashes before approving
   the upload. CPU CI success does not establish GPU correctness.
3. Approve the tag-triggered TestPyPI upload and verify its downloads against
   the recorded hashes.
4. Dispatch `publish.yml` from `main` with the same `version`, `target=pypi`,
   and the original release `run-id`. This promotes the identical files and
   requires a separate approval from a non-operator reviewer on `pypi`.

Publishing runs for one version and destination are serialized. Cancel an
existing waiting run before dispatching a replacement for that destination.

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
