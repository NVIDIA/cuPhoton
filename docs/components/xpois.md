# XPOIS

`cuphoton.xpois` fits PSF-matching kernels and differential backgrounds,
then subtracts a matched reference from a target image. It implements
Alard-Lupton-style Gaussian-polynomial bases for constant two-dimensional
kernels and a separable alternating solver. Its CLI group is
`cuphoton xpois`.

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

Automatic selection prefers CuPy, then Numba-CUDA, then CPU. cuTile is an
explicit experimental backend and is not selected by `auto`.

## Input and output contract

The reference and target are same-shaped 2D FITS or NPY images. The optional
CLI variance is target variance and must align with them. Optional masks must
also align. Kernels and auto-selected stamps use odd dimensions. See
[Data and artifact contracts](../data-artifacts.md#xpois-image-pairs).

A successful run persists `kernel.npy`, `matched.npy`, `residual.npy`,
`fit_mask.npy`, and `background.npy` plus `summary.json`. Auto-stamp fitting
also saves the selected-region metadata.

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
  --backend auto \
  --output-dir /path/to/runs
```

`fit-kernel` and `subtract` share the same solve but use different workflow
names. Use `--fit-mask` for a reviewed binary NPY selection, or
`--auto-stamp-mask` with an odd stamp size for compact-source selection.

Optional polynomial backgrounds and a flux-conserving basis rewrite are
available from the CLI. Inspect the fitted kernel sum, fit-pixel count,
chi-square, residual distribution, and source-scale residuals before accepting
a subtraction.

## Compare and review

Residual-hotspot and mask-component summaries group labeled foreground
pixels once, avoiding a full-image scan for every component. Component
statistics retain raster order and the input residual dtype, including
`mean_residual`. Equal peaks select the first pixel in raster order; equal
component ranks retain labeling order before `max_regions` truncation.

```bash
uv run cuphoton xpois benchmark-backends \
  --reference /path/to/reference.fits \
  --target /path/to/target.fits \
  --backends cpu,cupy,numba-cuda \
  --repeats 5 \
  --output-dir /path/to/runs

uv run cuphoton xpois evaluate-subtraction --run-dir /path/to/run
uv run cuphoton xpois review-bokeh --run-dir /path/to/run
```

Benchmark artifacts separate timings from numerical comparisons. Device work
is synchronized for measured iterations. Bokeh review is optional and can be
rebuilt from the numeric run.

## Separable-kernel Python API

The separable solver is available from the curated Python API even though the
main CLI workflow fits a constant two-dimensional kernel:

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

## Dragon image-pair batches

`fit-batch-dragon` distributes complete reference/target image-pair fits across
GPU workers. The coordinator reads a manifest, assigns one Dragon
ProcessGroup worker to each selected GPU, and balances work by input size.
Arrays and output files stay on shared storage; workers send compact result
records through Dragon queues. Each image-pair fit runs on one GPU.

### Runtime dependency

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

### Manifest and launch

Inputs and output directories must be accessible at the same paths on every
node. The command accepts a strict JSON or YAML manifest:

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

Loading the manifest records each unique input's resolved path, byte size,
and nanosecond modification time. These values are checked after coordinator
preflight and before and after each item. A change that preserves both size
and modification time is not detected; use immutable input data or external
checksums when content identity matters.

Launch through Dragon within an existing scheduler allocation:

```bash
.venv/bin/dragon examples/xpois/dragon_batch.py \
  --manifest /shared/manifests/image-pairs.yaml \
  --output-dir /shared/results/xpois-dragon \
  --name image-pairs-gpu4 \
  --max-workers 4 \
  --worker-timeout-sec 3600 \
  --backend cupy
```

The wrapper invokes `cuphoton xpois fit-batch-dragon`. This command requires
an explicit GPU backend: `cupy`, `numba-cuda`, or `cutile`. It rejects `auto`
instead of falling back to CPU when a GPU package is unavailable.

The coordinator enumerates actual `Node.gpus` IDs and selects workers
round-robin across hosts. Each worker checks its host and singleton
`CUDA_VISIBLE_DEVICES` assignment before importing the numerical backend.
Loopback hostname aliases are accepted only for a single-node allocation.

### Results and limits

Every attempt uses a new, immutable run directory. Each item has an atomic
terminal record under `records/`; an ordinary item error does not prevent
the remaining items in its shard from running. The final `summary.json`
checks for missing, duplicate, unexpected, malformed, or assignment-inconsistent
item and worker results, as well as nonzero worker exits. A worker that could
not write a terminal record still fails the run, but its declared errors are
reported under `shard_result_audit.write_failed_shards` rather than as an
evidence mismatch. Worker wall time
defaults to one hour. The coordinator attempts bounded stop and close cleanup
after failed starts or joins.

`coordinator_wall_sec` includes setup, worker cleanup, and terminal-result
checks. Writing the final summary falls outside that interval. Per-item
`timings_sec` separates input reads, preprocessing, solving, artifact writes,
review work, and other postprocessing; `wall_sec` retains the enclosing
workflow and item-runner measurements.

The executor does not retry failed items, recover dead workers, or split one
image-pair solve across GPUs. Inspect the terminal status before consuming a
run's outputs, and start a new run after a failed attempt.
