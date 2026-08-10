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

The reference and target are same-shaped 2D FITS or NPY images. Optional
variance and mask inputs must align with them. Kernels and auto-selected stamps
use odd dimensions. See [Data and artifact contracts](../data-artifacts.md#xpois-image-pairs).

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
