# XPOIS

`cuphoton.xpois` fits PSF-matching kernels and differential backgrounds,
then subtracts a matched reference from a target image. It implements
Alard-Lupton-style Gaussian-polynomial bases for constant two-dimensional
kernels, a global separable alternating solver, and a spatially varying
separable ALS model. An experimental research API jointly fits a spatial
Gaussian-polynomial kernel model. Its CLI group is `cuphoton xpois`.

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
uses the CPU reference implementation. Explicit `cupy` selection requires a
usable CuPy runtime and raises an error when that requirement is unmet. The
spatial Gaussian-polynomial research API follows the spatial-ALS rule.

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

Spatial ALS runs that reach the sweep budget persist their artifacts and
exit with status 0. The command prints a warning, and `summary.json` and
`evaluation.json` record `converged: false`. Raise `--als-iterations` or enable
`--flux-conserve`, then verify convergence before accepting the fit.

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
  --reference-hdu 1 --target-hdu 1 \
  --variance /path/to/target.fits --variance-hdu 3 \
  --mask-policy strict --reference-mask-hdu 2 --target-mask-hdu 2 \
  --auto-stamp-mask --auto-stamp-size 31 --auto-stamp-count 5 \
  --auto-peak-percentile 99.5 \
  --solver spatial-als \
  --backends cpu,cupy \
  --reference-backend cpu \
  --repeats 5 \
  --output-dir /path/to/runs

uv run cuphoton xpois evaluate-subtraction --run-dir /path/to/run
uv run cuphoton xpois review-bokeh --run-dir /path/to/run
```

`benchmark-backends` applies the same image masks, variance, crops, and fit
selection as `fit-kernel` and `subtract`. Explicit `--fit-mask` and
`--auto-stamp-mask` are mutually exclusive. With a crop, an explicit NPY mask
may have the full image shape or the cropped shape; the commands select the
same pixels from either form.

The summary records the effective `fit_pixel_count` per backend, fit-region
selection metadata, mask policy, and HDU/crop choices. Each backend saves its
effective fit mask as an NPY artifact. Report these settings and pixel counts
with speedups: whole-frame and compact-source fits measure different workloads.

`setup_timings.load_seconds` measures input reads and
`setup_timings.preprocess_seconds` measures masking and selection. Neither is
part of the repeated `solve_seconds` measurements, which include solving and
applying the kernel. Benchmark artifacts separate timings from numerical
comparisons. Device work is synchronized for measured iterations. Bokeh review is optional and can be
rebuilt from the numeric run.

## Separable-kernel Python APIs

The curated Python API provides a separable solver. The constant CLI model
uses a full two-dimensional kernel:

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
for both line profiles. Each axis has one profile, including when the basis
uses multiple Gaussian widths, so each realized kernel has rank at most one.
Shapes such as a sum of distinct circular Gaussians or a rotated anisotropic
PSF generally require a nonseparable model such as `solve_constant_kernel`.
The returned result exposes the local kernel through `kernel_at(y, x)`:

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

ALS means alternating linear least squares. The solver alternates linear
updates to the horizontal and vertical profiles using NumPy or CuPy. Callers
supply already registered images and an optional variance image. A fit mask
or explicit `(y, x)` sample positions can select the fit region; when
neither is supplied, all valid pixels in the centered-kernel interior are
fitted. Repeated positions are retained as multiplicity weights. Explicit
positions raise an error if any requested target, variance, or source
footprint is non-finite; mask and default selection use valid pixels. The
calling pipeline supplies camera calibration, PSF measurement, source
selection, astrometric registration, and unit interpretation, including
calibration-derived masks, variances, or fit samples. Flux conservation is
opt-in, matching the existing XPOIS CLI convention. When enabled, spatial
basis corrections are zero-sum and the signed kernel sum is one fitted,
position-independent scale. With flux conservation disabled, `flux_scale` is
the vertical reference multiplier; evaluate `kernel_at(y, x)` for the local
kernel sum and photometric scale. Near dependence between reference and
correction profiles can cause poor conditioning or slow convergence. Use
`--flux-conserve` (or `flux_conserve=True`) for the spatial model unless a
position-dependent kernel sum is required.

`SpatialALSConfig.tolerance` (`--als-tolerance`) is the relative
penalized-objective change that stops the alternating updates early. A
tolerance of `0` disables that stop: the fit runs to `max_iterations` or stops
at the numerical objective floor. Reaching that floor sets `converged` to
true; exhausting the iteration budget sets it to false.

The CuPy backend accelerates one spatial fit on one GPU. MPI and Dragon can
schedule those fits across complete image-pair items, but do not shard one fit
across GPUs. Keep single-fit solver time separate from orchestration and I/O
when comparing executors.

## Spatial Gaussian-polynomial research API

`solve_spatial_gaussian_polynomial_kernel` is an experimental CPU and CuPy
implementation of a spatial Gaussian-polynomial kernel model. Its fixed
Gaussian-polynomial basis, built from isotropic Gaussian envelopes,
represents nonseparable kernel modes. The ALS outer-product family also
represents kernels outside this fixed basis. The model follows the
spatial-kernel lineage of
[Alard (2000)](https://arxiv.org/abs/astro-ph/9903111), with independently
controlled photometric-scale, kernel-shape, and background fields as described
by [Bramich et al. (2013)](https://arxiv.org/abs/1210.2926). Integrations use the
array, coordinate, and weighting contracts described below. Import this
experimental API from its defining submodule:

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
    backend="cupy",
)
local_kernel = fit.kernel_at_local(local_y, local_x)
parent_kernel = fit.kernel_at_parent(detector_y, detector_x)
```

`SpatialGaussianPolynomialKernelFitSamples` can replace `fit_mask` when an
upstream source or stamp-selection stage needs to preserve exact pixel rows.
It accepts local `(y, x)` positions and optional positive
`relative_precision`. Each measurement must appear once; duplicate pixels
raise an error. The result reports `fit_objective`, the weighted residual sum
under the named `fit_weighting` policy.
`target_variance` weighting gives a chi-square-form objective; statistical
calibration also requires valid target variances and applicable model
assumptions.

Precisely, the minimized objective is
`J = sum_i relative_precision_i * residual_i**2 / target_variance_i`, with
each omitted factor set to one. Relative precision enters at its supplied
scale. Source variance contributes to diagnostics, and the nominal degrees of
freedom are the number of unique fitted pixels minus the number of fitted
parameters.

The model fits all fixed two-dimensional Gaussian-polynomial basis terms
jointly in one convex weighted linear solve. The first, unit-sum basis carries
the photometric-scale field; every remaining basis has zero sum and describes
kernel shape, preserving the kernel sum during shape variation. Both
backends use the same unregularized block QR reduction and raise an error for
ill-conditioned fits. Before the solve, a separate check rejects basis
kernels or design columns that cancel to rounding noise, such as a duplicated
component; this check complements the column-scaled condition number.
Chebyshev fields are evaluated in an explicit parent pixel-coordinate
domain. By default, each non-singleton input axis spans `[-1, 1]`; a singleton
axis maps to `0`. Supplying the crop origin and half-open parent bounding box
gives fitted coefficients the same coordinate interpretation across cutouts.
Callers associate this pixel domain with their detector and WCS metadata.
Result methods distinguish local-array from parent pixel coordinates
explicitly: local-frame evaluation is limited to the fitted array, while
parent-frame evaluation accepts coordinates between the first and last
pixel centers of `normalization_bbox`, inclusive. For a bounding box `(y0,
y1, x0, x1)`, this means `y0 <= y <= y1 - 1` and `x0 <= x <= x1 - 1`,
including fractional positions between centers. For a cutout it
extrapolates the fitted fields beyond the array within these bounds.

Access this experimental model through the Python API. The CPU path is the
same-model numerical reference; `backend="cupy"` executes the same
two-dimensional basis contraction and FP64 QR solve on one GPU, while
`backend="auto"` chooses CuPy when a usable CUDA device is available and CPU
otherwise.

`variance` supplies target-only weights for the fit objective. When
`source_variance` is also supplied, the result propagates independent source
pixel variances through each position-dependent fitted kernel and reports
`propagated_source_variance`, `marginal_residual_variance`, and
`marginal_standardized_residual`. These diagnostics use the coefficients from
the target-weighted fit and report NaN wherever a source-variance footprint is
missing. They describe marginal pixel variance for a fixed fitted model. The
residual retains cross-pixel correlations; a proper-difference or calibrated
detection-significance analysis requires additional treatment of those
correlations and fit uncertainty.

## Fixed-kernel marginal noise diagnostics

The survey-neutral Python API can propagate a reference variance plane through
an already fitted constant kernel and combine it with target variance:

```python
from cuphoton.xpois import (
    standardize_constant_kernel_residual,
    summarize_standardized_residuals,
)

noise = standardize_constant_kernel_residual(
    fit.residual,
    kernel=fit.kernel,
    target_variance=target_variance,
    reference_variance=reference_variance,
    valid_mask=evaluation_mask,
)
standardized = noise.standardized_residual
statistics = summarize_standardized_residuals(
    standardized[None, ...],
    valid_mask=noise.valid_mask[None, ...],
)
print(statistics.pixel_pooled.cardinal_lag1_rho)
```

This is a fixed-kernel marginal-diagonal calculation. The reference variance
is convolved with `kernel**2`; a non-unit kernel sum therefore applies its own
photometric scaling. The calculation uses the completed fit's residual and
leaves its weights and chi-square unchanged. The standardized residual
describes marginal noise under a fixed-kernel, independent-pixel model.
Whitening or calibrated significance requires additional treatment of
neighboring-pixel covariance, reference-target covariance, resampling
covariance, and fitted-kernel uncertainty. Prefer held-out pixels when
assessing fit quality.

`summarize_standardized_residuals` reports the RMS, the demeaned standard
deviation (the population root mean square of residuals centered on each
stamp's valid-pixel mean), and cardinal, diagonal, and radius-three lag
diagnostics for each stamp and for all valid pixels pooled together. The
pooled summary weights stamps by their valid pixels or lag-endpoint pairs; use
`per_stamp` when an equal-stamp aggregation is required. `radius3_rho_rms` is
an unweighted RMS of 24 noisy per-lag correlations, so white noise does not
drive it to zero. Gaussian white-noise simulations give a complete-stamp
floor of roughly 0.47 at the 4-by-4 minimum (where the corner lags rest on a
single pair), 0.16 at 8-by-8, and 0.03 at 32-by-32. For large stamps the
sampling contribution is about
`sqrt(mean(1 / N_lag))` over the per-lag pair counts. Per-stamp centering
also induces negative correlations: pooling many complete, independent
white-noise stamps with `N` pixels each makes each lag correlation approach
`-1 / (N - 1)`. The pooled correlation RMS therefore stays near 0.067 for
4-by-4 stamps even as the pair counts grow. Compare stamp sizes and masks
as well as pair counts, using a matching white-noise baseline. These
diagnostics do not whiten the residual, estimate a covariance model, set
acceptance thresholds, or calibrate statistical significance.

## Distributed image-pair batches

`fit-batch` assigns complete, independent reference/target pairs to GPU
workers. Each worker performs the complete kernel solve for its assigned pair
on one GPU. The two required selection flags name different layers:

| Flag | Selects | Values |
| --- | --- | --- |
| `--executor` | process placement, lifecycle, and result aggregation | `mpi`, `dragon` |
| `--backend` | the numerical fit implementation inside each worker | `cupy`, `numba-cuda`, `cutile` |

The command uses one solver and one set of fit options for every pair in a
batch. Select either the constant-kernel workflow or the spatial ALS workflow
for the whole run; per-pair solver overrides are not supported. For spatial
ALS, each worker solves one complete image pair on one GPU; the coefficient
solve is not distributed across workers.

The distributed command requires an explicit GPU backend from the table and
its runtime dependencies. `--backend auto`, CPU fallback, and options for a
different executor raise errors. Inspect the complete surface with:

```bash
uv run cuphoton xpois help fit-batch
```

### Runtime dependencies

The Dragon executor requires [DragonHPC](https://dragonhpc.github.io/dragon/doc/_build/html/index.html)
(Python distribution `dragonhpc`, import `dragon`) in the same Python
environment as cuPhoton on every participating node. Install DragonHPC
separately alongside cuPhoton's `gpu` extra and manage its version as an
external runtime dependency.

For example, install the released DragonHPC 0.14.2 package into a CUDA 13
cuPhoton environment:

```bash
uv sync --locked --python 3.12 --extra gpu
uv pip install --python .venv/bin/python "dragonhpc==0.14.2"
```

The [DragonHPC 0.14.2 wheels](https://pypi.org/project/dragonhpc/0.14.2/#files)
support CPython 3.11 through 3.13 on Linux x86-64 and AArch64 with glibc 2.28
or newer. Use one of those Python versions for this installation.

Use the installed `.venv/bin/dragon` launcher with the examples below.
`uv sync` removes packages outside the project lock, so repeat the DragonHPC
installation after resynchronizing the environment. Use the same cuPhoton
environment and DragonHPC version on every node. Each run records the
DragonHPC version it discovers. See the
[runtime notices](../../THIRD_PARTY_NOTICES.md#optional-distributed-runtime-inventory)
for licensing and installation details.

The MPI executor uses an external MPI or scheduler launcher. Collective
aggregation (`--aggregation-mode mpi`) also requires `mpi4py` built for the
selected MPI implementation. Shared-file aggregation exchanges results
through the filesystem, with the external launcher managing its processes.
Install the runtimes required by the selected executor on each node.

### Manifest and storage contract

Both executors consume the same strict JSON or YAML manifest and run the same
XPOIS item function. Manifests use resolved filesystem paths:

```yaml
schema: cuphoton.xpois.image-pairs/v2
pairs:
  - id: detector-0001
    reference: /shared/input/reference-0001.fits
    target: /shared/input/target-0001.fits
    reference_hdu: 1
    target_hdu: 1
    variance: /shared/input/variance-0001.fits
    variance_hdu: 1
    fit_positions: /shared/input/fit-positions-0001.npy
```

`fit_positions` is optional and spatial-ALS-only. It contains post-crop
integer `(y, x)` rows; repeated rows remain repeated so overlapping source
stamps retain their multiplicity weighting. Version 2 adds this field; version
1 manifests remain accepted and retain their original canonical hash.

Within a spatial-ALS batch, different pairs can use `fit_mask` or
`fit_positions`, but a pair cannot supply both. Preflight reads each position
file to validate its dtype and coordinates before item execution; the assigned
worker reads it again for the fit. Position arrays are not cached in the
coordinator or sent through Dragon queues.

Inputs and output roots must be visible at the same paths on every node. Every
rank must resolve `--output-dir/--name` to the same final run directory.
Loading a manifest captures each unique input's resolved path, byte size, and
nanosecond modification time. That identity is checked during preflight and
before and after each item. The manifest SHA-256 binds both the canonical pair
entries and this captured identity, and MPI ranks require the same digest
before run setup. This metadata-based identity check scales to multi-gigabyte
inputs. Use an immutable dataset version or an external content checksum to
bind the run to exact input contents.

Work is assigned as deterministic input-byte-balanced whole-item shards.
Scientific arrays stay on shared storage; only compact rank or worker results
are aggregated. Runs are immutable. Use a new `--name` for each new attempt.
The marker-only recovery retry described below must reuse the existing
`--name` and `--attempt-id`.

Spatial ALS supports only `cupy` in this GPU-only command. Selecting
`numba-cuda` or `cutile` with `--solver spatial-als` fails before either executor
launches. Both executor routes accept `--solver spatial-als --spatial-degree 2`
to select the spatial model.

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
then executes the requested command. Place the helper immediately after the
launcher, before any shell or Python process.

Rank zero opens and validates every manifest input before execution. Other
ranks load the manifest and stat its inputs to agree on the same digest and
assignment through metadata checks. Collective consensus broadcasts the
root-validated digest; file mode publishes it in the ready marker. A root
preflight failure prevents either mode from starting work. Each rank still
verifies input identity when executing its assigned items.

Collective startup errors are exchanged before rank-context setup, and every
rank verifies a root-written nonce through the resolved run directory before
GPU work starts. The launcher owns termination for MPI import and
initialization failures as well as later process exits. Configure it to
terminate the remaining collective ranks when any task exits; with
Slurm use `srun --kill-on-bad-exit=1` (or the site's equivalent) for
`--aggregation-mode mpi`. Open MPI must retain its fail-fast policy for a
nonzero or lost rank. The launcher also handles signals, process aborts, and
failures before MPI initialization.

`--aggregation-mode mpi` requires a compatible `mpi4py` installation.
`--rank-setup-timeout-sec` bounds shared-filesystem metadata handoffs: setup
markers, collective rank/record visibility before terminal audit, and staged
file-rank visibility before promotion. It applies to both aggregation modes,
defaults to 600 seconds, and must be identical on every rank. The launcher's
fail-fast policy governs MPI collectives. Each metadata phase has one shared
absolute budget covering all ranks and artifacts.

`--aggregation-mode files` instead exchanges rank metadata through the shared
output filesystem; give every rank the same explicit `--name` and one unique
shared `--attempt-id` for that launch. File-mode ownership additionally binds
ranks to the launcher's PMIx namespace or direct Slurm job and numeric step.
A nested `mpirun` uses its PMIx namespace. For other launchers, set a fresh
token before each launcher invocation and propagate it unchanged to every
rank:

```bash
CUPHOTON_MPI_LAUNCH_ID="$(python -c 'import uuid; print(uuid.uuid4().hex)')" \
  mpirun -x CUPHOTON_MPI_LAUNCH_ID ...
```

Use one shared token per launch. Launch ownership isolates rank staging and
preflight records. `--rank-timeout-sec` applies to file aggregation,
defaults to 3600 seconds, and must also be identical on every rank. It
bounds rank zero's wait for peer completion markers after rank zero finishes
its own shard, so size it above the worst expected completion skew between
rank zero and the slowest peer. The default aggregation mode is `mpi`;
select `files` explicitly for file aggregation. Rank zero claims the attempt
identity atomically and rejects reused IDs. Recovery supports an exact retry
after `summary.json` committed but the terminal attempt-marker write failed.
Rank zero validates the regular marker, ready record, run record, summary,
manifest, options, topology, and timeouts, then repairs only that marker and
directs the operator to the existing immutable summary. Interrupted attempts
without a committed summary are retained for inspection; restart with a new
`--name` and `--attempt-id`. Operators control cleanup of this retained
evidence. The output root's `.mpi-attempts/` directory holds the atomic
attempt marker and retained per-rank preflight, staging, and completion
evidence outside the immutable run directory. File-mode ranks publish their
completion marker last. Before any promotion, the coordinator requires
consistent completion and rank status, regular JSON evidence, and real local
output trees for successful items; symlinks and missing success artifacts
stay in staging and fail the run. In both executors, a failed item can
retain partial output under `items/`. Consumers must check its terminal
record before using that output. The coordinator promotes artifacts from a
completed, validated rank into the immutable run directory before writing
its summary. Results arriving after the timeout remain outside the audited
run. If a shared-filesystem rename fails mid-promotion, the failed summary
records `PartialRankPromotion` and lists the paths already published.

With file aggregation, rank zero owns the evidence timeout, terminal batch
status, and authoritative launcher exit code. Nonzero ranks return zero after
publishing their completion markers, even when their local shard failed, so
rank zero can finish collecting and persisting the launch-wide failure
evidence. With collective aggregation, rank zero broadcasts the terminal
decision and a failed batch raises consistently on every rank.
File aggregation requires a launch policy that lets rank zero finish the
audit after peers publish their evidence and return. Peer processes and their
GPU allocations remain active after a file-mode evidence timeout. Configure a
scheduler wall-time limit or use job-level cancellation for hung ranks; the
launcher owns remote-process termination and may wait for those ranks after
rank zero has exited.

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

The manifest must contain at least one image pair per task. An empty MPI shard
raises an error before execution.

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
directory, summary path, and terminal status. Durable per-item records and
the final summary audit missing, duplicate, unexpected, malformed, failed,
and assignment-inconsistent results. After an ordinary item error, the
worker continues with the remaining items in its shard. Any item, rank,
worker, or audit failure makes the batch command return nonzero after
evidence is persisted. Declared worker record-write errors still fail the
run and are reported under `shard_result_audit.write_failed_shards`,
separately from evidence mismatches. The MPI rank-result audit reports
trustworthy workload and setup failures in separate fields, apart from
malformed or identity-inconsistent rank evidence. Valid record-write
failures appear in `rank_result_audit.write_failed_ranks` and fail the run
even when a record became visible before durability failed. GPU identity
comparison prefers UUID when both peers report one and otherwise uses PCI
identity only on the same host. Partial lookup failures are retained as
warnings when a stable identifier survives, while an incomparable same-host
pair still fails closed.

For an MPI/Dragon comparison, stage one immutable manifest before timing and
hold the allocation, filesystem, cache policy, cuPhoton revision, numerical
backend, and fit options constant. Rotate launch order and compare exact output
artifacts, exactly-once records, GPU placement, clean exits, complete launcher
wall time, coordinator or rank work time, and per-item phase timings.

Select and configure the launcher before each batch. Each GPU processes
complete image pairs. After failed work or a dead process, start a new run
and inspect its terminal status before consuming outputs. The exact
marker-recovery retry described above reuses a completed run's summary.
