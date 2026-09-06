# XPOIS

`cuphoton.xpois` fits PSF-matching kernels and differential backgrounds,
then subtracts a matched reference from a target image. It implements
Alard-Lupton-style Gaussian-polynomial bases for constant two-dimensional
kernels, a global separable alternating solver, and a spatially varying
separable ALS model. Its CLI group is `cuphoton xpois`.

## Install and smoke test

```bash
# CUDA 13
uv sync --locked --extra dev --extra gpu --extra viz
uv run python examples/run_quickstarts.py --component xpois --require-gpu

# CPU
uv sync --locked --extra dev --extra viz
uv run python examples/run_quickstarts.py \
  --component xpois --profile cpu
```

Backend selection is solver-specific. For the constant solver, `auto` prefers
CuPy, then Numba-CUDA, then CPU; cuTile remains explicit and experimental. For
spatial ALS, `auto` prefers CuPy when it has a usable CUDA device and otherwise
uses the CPU reference implementation. Explicit `cupy` selection fails instead
of falling back when its runtime is unavailable.

## Input and output contract

The reference and target are same-shaped 2D FITS or NPY images. The optional
CLI variance is target variance and must align with them. Optional masks must
also align. Kernels and auto-selected stamps use odd dimensions. See
[Data and artifact contracts](../data-artifacts.md#xpois-image-pairs).

A successful run persists `matched.npy`, `residual.npy`, `fit_mask.npy`, and
`background.npy` plus `summary.json`. Constant-kernel fits save `kernel.npy`.
Spatial ALS fits instead save the line bases and coefficient fields plus
`kernel_center.npy`, a clearly named preview evaluated at the image center.
Auto-stamp fitting also saves the selected-region metadata.

## Fit and subtract

```bash
uv run cuphoton xpois fit-kernel \
  --reference /path/to/reference.fits \
  --target /path/to/target.fits \
  --variance /path/to/variance.fits \
  --auto-stamp-mask \
  --backend auto \
  --output-dir /path/to/runs

uv run cuphoton xpois subtract \
  --reference /path/to/reference.fits \
  --target /path/to/target.fits \
  --solver spatial-als \
  --spatial-degree 2 \
  --flux-conserve \
  --backend cupy \
  --output-dir /path/to/runs
```

`fit-kernel` and `subtract` share the same solve but use different workflow
names. `--solver constant` remains the default. Use `--fit-mask` for a
reviewed binary NPY selection, or `--auto-stamp-mask` with an odd stamp size
for compact-source selection.

Spatial ALS runs that reach the sweep budget still persist their artifacts
and exit with status 0, but the command prints a warning and `summary.json`
and `evaluation.json` record `converged: false`; do not accept such a run
without raising `--als-iterations` or enabling `--flux-conserve`.

Optional polynomial backgrounds and a flux-conserving basis rewrite are
available from the CLI. Inspect the fitted kernel sum (`kernel_sum_center`
for spatial ALS), fit-pixel count, chi-square, residual distribution, and
source-scale residuals before accepting a subtraction.

## Compare and review

Both fitting commands generate numeric hotspot metadata and, when Bokeh is
installed, an interactive HTML review by default. Add `--no-review` to skip
these artifacts and their preparation. The summary records
`review_enabled=false` and retains the fitted arrays, fit metrics, mask
metadata, and input paths. Keep the input files available to generate the
HTML later with `review-bokeh --run-dir /path/to/run`.

For large frames, including 4096 by 4096 images, inspect the summary's
`review_generation_and_write_sec` separately from `solve_sec`: review work
can exceed the solve time. Automatic stamp selection also processes the
image and contributes to `preprocess_sec`; `--no-review` still performs that
selection when requested. Hotspot significance uses the robust residual
noise measured inside the fit region, so changing the fit mask changes the
noise unit used by the review.

Residual-hotspot and mask-component summaries group labeled foreground
pixels once, avoiding a full-image scan for every component. Component
statistics retain raster order and the input residual dtype, including
`mean_residual`. Equal peaks select the first pixel in raster order; equal
component ranks retain labeling order before `max_regions` truncation.

```bash
uv run cuphoton xpois benchmark-backends \
  --reference /path/to/reference.fits \
  --target /path/to/target.fits \
  --solver spatial-als \
  --backends cpu,cupy \
  --reference-backend cpu \
  --repeats 5 \
  --output-dir /path/to/runs

uv run cuphoton xpois evaluate-subtraction --run-dir /path/to/run
uv run cuphoton xpois review-bokeh --run-dir /path/to/run
```

Benchmark artifacts separate timings from numerical comparisons. Device work
is synchronized for measured iterations. Bokeh review is optional and can be
rebuilt from the numeric run.

## Separable-kernel Python APIs

The separable solver is available from the curated Python API even though the
constant CLI model is a full two-dimensional kernel:

```python
from cuphoton.xpois import GaussianBasisComponent, solve_separable_kernel

components = (
    GaussianBasisComponent(sigma=1.5, degree=2),
    GaussianBasisComponent(sigma=3.0, degree=1),
)
fit = solve_separable_kernel(
    reference,
    target,
    components,
    variance=variance,
)
print(fit.converged, fit.iterations, fit.kernel.shape)
```

`solve_constant_kernel`, Gaussian-basis builders, background helpers, stamp
helpers, and the result dataclasses are also exported from
`cuphoton.xpois`.

`solve_spatial_als` fits a different model. At pixel `p`, its kernel is the
outer product `K_p(v, u) = V_p(v) H_p(u)`, with Chebyshev coefficient fields
for both line profiles. There is only one profile per axis, even with
multiple Gaussian widths, so each realized kernel has rank at most one. A
sum of distinct circular Gaussians or a rotated anisotropic PSF is generally
not representable; the kernel-shape expressivity differs from the
nonseparable `solve_constant_kernel` model. The returned result exposes
`kernel_at(y, x)` rather than claiming one kernel represents the full image:

```python
from cuphoton.xpois import SpatialALSConfig, solve_spatial_als

fit = solve_spatial_als(
    reference,
    target,
    components,
    variance=variance,
    fit_mask=fit_mask,
    config=SpatialALSConfig(spatial_degree=2, flux_conserve=True),
    backend="auto",
)
center_kernel = fit.kernel_at(
    (reference.shape[0] - 1) / 2,
    (reference.shape[1] - 1) / 2,
)
```

`backend="cpu"` selects the NumPy reference implementation,
`backend="cupy"` requires a usable CUDA device, and `backend="auto"` uses CuPy
when available before falling back to CPU. Inputs and returned arrays remain
NumPy host arrays, and both implementations perform the fit in FP64.

Each kernel axis must have at least as many pixels as the line basis has
functions; the CLI default basis has six, so the spatial model needs
`--kernel-height` and `--kernel-width` of at least 7, and the solver rejects
smaller kernels with a message naming the first failing axis, its length,
and the number of basis functions.

ALS means alternating linear least squares. This model is unrelated to
Levenberg-Marquardt fitting and does not use the third-party `lmfit` package.
The solver is instrument-neutral. Callers supply already registered images and
an optional variance image. A fit mask or explicit `(y, x)` sample positions
can select the fit region; when neither is supplied, all valid pixels in the
centered-kernel interior are fitted. Repeated positions are retained as
multiplicity weights. Explicit positions fail closed if any requested target,
variance, or source footprint is non-finite; mask and default selection omit
invalid pixels. Camera calibration, PSF measurement, source selection,
astrometric registration, and unit interpretation remain responsibilities of
the calling pipeline. The solver embeds no camera calibration or observational
data; callers provide any calibration-derived masks, variances, or fit samples.
Flux conservation is opt-in, matching the existing XPOIS CLI convention. When
enabled, spatial basis corrections are zero-sum and the signed kernel sum is
one fitted, position-independent scale. Without it, `flux_scale` is the
vertical reference multiplier rather than a standalone photometric scale;
evaluate `kernel_at(y, x)` for the local kernel sum. Near dependence between
reference and correction profiles can cause poor conditioning or slow
convergence. Use `--flux-conserve` (or `flux_conserve=True`) for the spatial
model unless a position-dependent kernel sum is required.

Setting `tolerance=0` disables the relative-change stopping criterion.
Only the numerical objective floor can end iterations before the full
`max_iterations` budget.
`SpatialALSConfig.tolerance`
(`--als-tolerance`) is the relative penalized-objective change that stops the
alternating updates early. A tolerance of `0` disables that stop; the fit then
runs to `max_iterations` unless the objective reaches its numerical floor, and
`converged` is otherwise false.

The CuPy backend accelerates one spatial fit on one GPU. MPI and Dragon are
separate whole-item orchestration layers; enabling this backend does not shard
one fit or add the spatial solver to those batch paths.

## Spatial Gaussian-polynomial research API

`solve_spatial_gaussian_polynomial_kernel` is an experimental CPU reference for
a less restrictive model than rank-one ALS. It uses fixed two-dimensional
Gaussian-polynomial basis kernels in the spatial-kernel lineage of
[Alard (2000)](https://arxiv.org/abs/astro-ph/9903111), with independently
controlled photometric-scale, kernel-shape, and background fields as described
by [Bramich et al. (2013)](https://arxiv.org/abs/1210.2926). This identifies the
model family without claiming compatibility with a legacy or survey pipeline.
While experimental, import it from its defining submodule rather than the
curated `cuphoton.xpois` root API:

```python
from cuphoton.xpois.spatial_gaussian_polynomial import (
    SpatialKernelDomain,
    SpatialGaussianPolynomialKernelConfig,
    solve_spatial_gaussian_polynomial_kernel,
)

fit = solve_spatial_gaussian_polynomial_kernel(
    reference,
    target,
    components,
    variance=target_variance,
    source_variance=reference_variance,
    fit_mask=fit_mask,
    spatial_domain=SpatialKernelDomain(
        array_origin_yx=(crop_y0, crop_x0),
        normalization_bbox=(0, detector_height, 0, detector_width),
    ),
    config=SpatialGaussianPolynomialKernelConfig(
        shape_degree=2,
        photometric_degree=0,
        background_degree=1,
    ),
)
local_kernel = fit.kernel_at_local(local_y, local_x)
parent_kernel = fit.kernel_at_parent(detector_y, detector_x)
```

`SpatialGaussianPolynomialKernelFitSamples` can replace `fit_mask` when an
upstream source or
stamp-selection stage needs to preserve exact pixel rows. It accepts local
`(y, x)` positions and optional positive `relative_precision`; duplicate pixels
are rejected because repeating one measurement does not create independent
information. The result reports `fit_objective`, the weighted residual sum
under the named `fit_weighting` policy, rather than a `chi2` field; it equals
a chi-square only under `target_variance` weighting.

Precisely, the minimized objective is
`J = sum_i relative_precision_i * residual_i**2 / target_variance_i`, with
each omitted factor set to one. Relative precision is not normalized, source
variance remains diagnostic-only, and the nominal degrees of freedom are the
number of unique fitted pixels minus the number of fitted parameters.

The model fits all fixed two-dimensional Gaussian-polynomial basis terms
jointly in one convex weighted linear solve. The first, unit-sum basis carries
the photometric-scale field; every remaining basis has zero sum and describes
kernel shape. This prevents shape variation from silently changing the kernel
sum. A block QR reduction avoids forming normal equations in the CPU reference,
and ill-conditioned fits fail rather than acquiring an implicit ridge model.
Basis kernels or design columns that cancel to rounding noise, such as a
duplicated component, are rejected before the solve because the column-scaled
condition number cannot detect them.
Chebyshev fields are evaluated in an explicit parent pixel-coordinate domain.
If no domain is supplied, the complete input array spans `[-1, 1]` on both
axes. Supplying the crop origin and half-open parent bounding box gives fitted
coefficients the same coordinate interpretation across cutouts; it does not
identify a detector or WCS. Result methods distinguish local-array from parent
pixel coordinates explicitly: local-frame evaluation is limited to the fitted
array, while parent-frame evaluation is defined anywhere inside
`normalization_bbox`, so for a cutout it extrapolates the fitted fields beyond
the array.

This experimental Python API has no CLI selector or GPU backend.

`variance` supplies target-only weights for the fit objective. When
`source_variance` is also supplied, the result propagates independent source
pixel variances through each position-dependent fitted kernel and reports
`propagated_source_variance`, `marginal_residual_variance`, and
`marginal_standardized_residual`. Source variance is diagnostic-only: missing
source-variance footprints do not alter the fitted coefficients, but the
corresponding diagnostic pixels are NaN. These products do not include fit
uncertainty or cross-pixel covariance. The residual therefore remains an
ordinary correlated difference image, not a proper-difference or calibrated
detection-significance image. This API does not decorrelate the residual.

## Fixed-kernel marginal noise diagnostics

The survey-neutral Python API can propagate a reference variance plane through
an already fitted constant kernel and combine it with target variance:

```python
from cuphoton.xpois import standardize_constant_kernel_residual

noise = standardize_constant_kernel_residual(
    fit.residual,
    kernel=fit.kernel,
    target_variance=target_variance,
    reference_variance=reference_variance,
    valid_mask=evaluation_mask,
)
standardized = noise.standardized_residual
```

This is a fixed-kernel marginal-diagonal calculation. The reference variance
is convolved with `kernel**2`; a non-unit kernel sum therefore applies its own
photometric scaling. It does not change fit weights or chi-square and does not
represent neighboring-pixel covariance, reference-target covariance,
resampling covariance, or fitted-kernel uncertainty. The standardized residual
is a descriptive diagnostic, not a whitened residual or calibrated
significance image. Prefer held-out pixels when assessing fit quality.

## Distributed image-pair batches

`fit-batch` assigns complete, independent reference/target pairs to GPU
workers. It does not split one image or one kernel solve across devices. The
two required selection flags name different layers:

| Flag | Selects | Values |
| --- | --- | --- |
| `--executor` | process placement, lifecycle, and result aggregation | `mpi`, `dragon` |
| `--backend` | the numerical fit implementation inside each worker | `cupy`, `numba-cuda`, `cutile` |

The distributed command deliberately rejects `--backend auto` and CPU
fallback. A missing GPU package therefore cannot silently change a distributed
run's execution mode. Executor-specific options are also rejected when used
with the other executor. Inspect the complete surface with:

```bash
uv run cuphoton xpois help fit-batch
```

This command runs the constant-kernel workflow. It is whole-image-pair
orchestration, not a distributed implementation of the spatial ALS model.

### Runtime dependencies

The Dragon executor requires [DragonHPC](https://dragonhpc.github.io/dragon/doc/_build/html/index.html)
(Python distribution `dragonhpc`, import `dragon`) in the same Python
environment as cuPhoton on every participating node. Installing cuPhoton,
including its `gpu` extra, does not install DragonHPC. DragonHPC is installed
separately and is not included in cuPhoton's dependency lock or wheel.

For example, install the released DragonHPC 0.14.2 package into a CUDA 13
cuPhoton environment:

```bash
uv sync --locked --python 3.12 --extra gpu
uv pip install --python .venv/bin/python "dragonhpc==0.14.2"
```

The [DragonHPC 0.14.2 wheels](https://pypi.org/project/dragonhpc/0.14.2/#files)
support CPython 3.11 through 3.13 on Linux x86-64 and AArch64 with glibc 2.28
or newer. Use one of those Python versions for this installation; a CPython
3.14 wheel is not published for this DragonHPC release.

Use the installed `.venv/bin/dragon` launcher with the examples below.
`uv sync` removes packages outside the project lock, so repeat the DragonHPC
installation after resynchronizing the environment. Use the same cuPhoton
environment and DragonHPC version on every node. Each run records the
DragonHPC version it discovers. See the
[runtime notices](../../THIRD_PARTY_NOTICES.md#optional-distributed-runtime-inventory)
for licensing and installation details.

The MPI executor uses an external MPI or scheduler launcher. Collective
aggregation (`--aggregation-mode mpi`) also requires `mpi4py` built for the
selected MPI implementation. Shared-file aggregation does not import
`mpi4py`. Install the runtimes required by the selected executor on each node;
the MPI executor does not require DragonHPC.

### Manifest and storage contract

Both executors consume the same strict JSON or YAML manifest and run the same
XPOIS item function. Manifests use resolved filesystem paths:

```yaml
schema: cuphoton.xpois.image-pairs/v1
pairs:
  - id: detector-0001
    reference: /shared/input/reference-0001.fits
    target: /shared/input/target-0001.fits
    reference_hdu: 1
    target_hdu: 1
    variance: /shared/input/variance-0001.fits
    variance_hdu: 1
```

Inputs and output roots must be visible at the same paths on every node. Every
rank must resolve `--output-dir/--name` to the same final run directory.
Loading a manifest captures each unique input's resolved path, byte size, and
nanosecond modification time. That identity is checked during preflight and
before and after each item. The manifest SHA-256 binds both the canonical pair
entries and this captured identity, and MPI ranks require the same digest
before run setup. It avoids reading multi-gigabyte inputs solely to hash them,
but is not a content checksum: use an immutable dataset version or an external
checksum when content identity matters.

Work is assigned as deterministic input-byte-balanced whole-item shards.
Scientific arrays stay on shared storage; only compact rank or worker results
are aggregated. Runs are immutable. Use a new `--name` for each new attempt.
The marker-only recovery retry described below must reuse the existing
`--name` and `--attempt-id`.

### Launch with MPI

An external `mpirun` or Slurm `srun` establishes the rank topology. With Open
MPI, use the installed `cuphoton-openmpi-rank-exec` helper so every child
narrows `CUDA_VISIBLE_DEVICES` before Python, `mpi4py`, or a CUDA-aware MPI can
initialize CUDA:

```bash
: "${CUDA_VISIBLE_DEVICES:?must enumerate the allocated GPUs}"
mpirun -n 4 \
  --map-by slot \
  --bind-to none \
  -x CUDA_VISIBLE_DEVICES \
  .venv/bin/cuphoton-openmpi-rank-exec -- \
  .venv/bin/cuphoton xpois fit-batch \
  --executor mpi \
  --backend cupy \
  --manifest /shared/manifests/fixed-32.yaml \
  --output-dir /shared/results/xpois-mpi \
  --name fixed-32-mpi4 \
  --aggregation-mode mpi
```

The parent visibility mask must enumerate the allocation in local-rank order.
The helper validates that mask and Open MPI's local rank metadata, preserves
the allocation in `CUPHOTON_ALLOCATED_CUDA_VISIBLE_DEVICES`, selects one token,
and only then executes the requested command. Do not place another shell or
Python process before the helper.

Rank zero opens and validates every manifest input before execution. Other
ranks load the manifest and stat its inputs to agree on the same digest and
assignment, without repeating the full input preflight. Collective consensus
broadcasts the root-validated digest; file mode publishes it in the ready
marker. A root preflight failure prevents either mode from starting work.
Each rank still verifies input identity when executing its assigned items.

Collective startup errors are exchanged before rank-context setup, and every
rank verifies a root-written nonce through the resolved run directory before
GPU work starts. A rank that cannot import or initialize MPI has no
communicator through which cuPhoton can report its failure. The launcher must
therefore terminate the remaining collective ranks when any task exits; with
Slurm use `srun --kill-on-bad-exit=1` (or the site's equivalent) for
`--aggregation-mode mpi`. Open MPI must retain its fail-fast policy for a
nonzero or lost rank. cuPhoton deliberately leaves job-wide termination to
the launcher: a Python exception hook cannot handle signals, process aborts,
or a rank that never initializes MPI.

`--aggregation-mode mpi` requires a compatible `mpi4py` installation.
`--rank-setup-timeout-sec` bounds shared-filesystem metadata handoffs: setup
markers, collective rank/record visibility before terminal audit, and staged
file-rank visibility before promotion. It applies to both aggregation modes,
defaults to 600 seconds, and must be identical on every rank. It does not bound
MPI collectives themselves; retain the launcher's fail-fast policy described
above. Each metadata phase has one shared absolute budget; the timeout is not
restarted for every rank or artifact.

`--aggregation-mode files` instead exchanges rank metadata through the shared
output filesystem; give every rank the same explicit `--name` and one unique
shared `--attempt-id` for that launch. File-mode ownership additionally binds
ranks to the launcher's PMIx namespace or direct Slurm job and numeric step.
A nested `mpirun` uses its PMIx namespace, not the surrounding Slurm allocation.
For a launcher without these identifiers, set a fresh token **before** each
launcher invocation and propagate it unchanged to every rank:

```bash
CUPHOTON_MPI_LAUNCH_ID="$(python -c 'import uuid; print(uuid.uuid4().hex)')" \
  mpirun -x CUPHOTON_MPI_LAUNCH_ID ...
```

Do not generate this token separately in each rank or reuse it across launches.
Competing launches cannot publish into each other's rank staging or preflight
records. `--rank-timeout-sec` applies only to
file aggregation, defaults to 3600 seconds, and must also be identical on
every rank. It bounds rank zero's wait for peer completion markers after rank
zero finishes its own shard, so size it above the worst expected completion
skew between rank zero and the slowest peer. The default aggregation mode is
`mpi`; there is no silent fallback to file aggregation. Rank zero claims the
attempt identity atomically, so a reused ID is rejected instead of overwriting
another launch. The only recovery exception is an exact retry after
`summary.json` committed but the terminal attempt-marker write failed. Rank
zero validates the regular marker, ready record, run record, summary, manifest,
options, topology, and timeouts, then repairs only that marker and directs the
operator to the existing immutable summary. Interrupted attempts without a
committed summary are retained for inspection; restart with a new `--name` and
`--attempt-id`. They are never reclaimed automatically.
The output root's `.mpi-attempts/` directory holds the atomic attempt marker
and retained per-rank preflight, staging, and completion evidence outside the
immutable run directory. File-mode ranks publish their completion marker last.
Before any promotion, the coordinator requires consistent completion and rank
status, regular JSON evidence, and real local output trees for successful
items; symlinks and missing success artifacts stay in staging and fail the run.
In both executors, a failed item can retain partial output under `items/`.
Consumers must check its terminal record before using that output.
It promotes artifacts only from a completed, validated rank into the immutable
run directory before writing its summary, so a rank that finishes after a
timeout cannot change the audited run. If a shared-filesystem rename fails
mid-promotion, the failed summary records `PartialRankPromotion` with the paths
already published; it never claims an all-or-nothing rank publication.

With file aggregation, rank zero owns the evidence timeout, terminal batch
status, and authoritative launcher exit code. Nonzero ranks return zero after
publishing their completion markers, even when their local shard failed, so
rank zero can finish collecting and persisting the launch-wide failure
evidence. With collective aggregation, rank zero broadcasts the terminal
decision and a failed batch raises consistently on every rank.
Do not add collective fail-fast launch policy to file aggregation: nonzero
ranks intentionally publish their evidence and return so rank zero can finish
the audit. A file-mode evidence timeout does not cancel peers or release
their GPUs. Configure a scheduler wall-time limit or use job-level cancellation
for hung ranks; rank zero cannot safely terminate remote launcher-owned
processes. A launcher may wait for those ranks after rank zero has exited.

When Slurm already gives each task singleton GPU visibility, it can launch the
unified command directly:

```bash
srun --nodes=2 \
  --ntasks=16 \
  --ntasks-per-node=8 \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  .venv/bin/cuphoton xpois fit-batch \
  --executor mpi \
  --backend cupy \
  --manifest /shared/manifests/fixed-32.yaml \
  --output-dir /shared/results/xpois-mpi \
  --name fixed-32-mpi16 \
  --aggregation-mode files \
  --attempt-id fixed-32-mpi16-attempt-1
```

The manifest must contain at least one image pair per task. XPOIS rejects idle
MPI ranks instead of launching ranks with empty shards.

### Launch with Dragon

The checked-in wrapper routes the same `fit-batch` command with
`--executor dragon`. The following launch selects TCP explicitly for
both infrastructure and overlay transport. Verify transport availability
for your installation before launching a distributed workload:

```bash
.venv/bin/dragon -m -N 2 -w slurm -t tcp -o tcp \
  examples/xpois/dragon_batch.py \
  --backend cupy \
  --manifest /shared/manifests/fixed-32.yaml \
  --output-dir /shared/results/xpois-dragon \
  --name fixed-32-dragon16 \
  --max-workers 16 \
  --worker-timeout-sec 3600
```

Here `-m` is Dragon's multi-node override. The Dragon coordinator enumerates
the allocation's actual hosts and GPU IDs, places one ProcessGroup worker per
selected GPU, and verifies singleton visibility before importing the numerical
backend. `--max-workers`, `--worker-timeout-sec`, and
`--result-timeout-sec` apply only to this executor.

For a single-node launch, use `-s` instead of `-m -N 2 -w slurm`, while
retaining `-t tcp -o tcp` to select TCP transport.

### Results and limits

Both routes print a compact result containing the executor, run ID, run
directory, summary path, and terminal status. Durable per-item records and the
final summary audit missing, duplicate, unexpected, malformed, failed, and
assignment-inconsistent results. An ordinary item error does not prevent the
remaining items in its shard from running, but any item, rank, worker, or audit
failure makes the batch command return nonzero after evidence is persisted.
Declared worker record-write errors still fail the run and are reported
under `shard_result_audit.write_failed_shards` rather than as evidence
mismatches.
The MPI rank-result audit reports trustworthy workload and setup failures in
separate fields, apart from malformed or identity-inconsistent rank evidence.
Valid record-write failures appear in `rank_result_audit.write_failed_ranks`
and fail the run even when a record became visible before durability failed.
GPU identity comparison prefers UUID when both peers report one and otherwise
uses PCI identity only on the same host. Partial lookup failures are retained
as warnings when a stable identifier survives, while an incomparable
same-host pair still fails closed.

For an MPI/Dragon comparison, stage one immutable manifest before timing and
hold the allocation, filesystem, cache policy, cuPhoton revision, numerical
backend, and fit options constant. Rotate launch order and compare exact output
artifacts, exactly-once records, GPU placement, clean exits, complete launcher
wall time, coordinator or rank work time, and per-item phase timings.

This interface does not retry failed work, resume after a dead process,
split one image across GPUs, or select a launcher automatically. Start a
new run after a failed attempt and inspect the terminal status before
consuming its outputs.
