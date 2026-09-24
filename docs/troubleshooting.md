# Troubleshooting

## Confirm the environment first

Run commands through the environment created for this checkout:

```bash
uv run python -c 'import sys, cuphoton; print(sys.executable, cuphoton.__version__)'
uv lock --check
uv run cuphoton xray doctor
```

If an executable is missing, rerun `uv sync` with the required extra. If the
import resolves to another checkout, inspect `sys.executable`, `cuphoton.__file__`,
and any `PYTHONPATH` entries before debugging the workflow.

## A GPU run used CPU

The default profile may fall back to CPU. Check the workflow or quickstart
summary for the resolved backend and device. Then verify that:

- the environment was synced with `--extra gpu`;
- the NVIDIA driver is visible through `nvidia-smi`;
- PyTorch reports `torch.cuda.is_available()`;
- CuPy can allocate and synchronize a small array; and
- `CUDA_VISIBLE_DEVICES` includes the intended GPU, if set.

Use `--require-gpu` in the synthetic runner to require a CUDA run. Select the
`cutile` backend explicitly with a compatible Python, `cuda-tile` runtime,
and TileIR compiler.

## CUDA package or driver mismatch

cuPhoton's GPU dependencies target CUDA 13. Recreate environments with mixed
CUDA 12/13 packages from `uv.lock`. Install a compatible NVIDIA driver as well
as the CUDA runtime dependencies. Record the driver, GPU, Python, and resolved
package versions in bug reports.

## A command rejected the input

Use the component inspection or validation command before the expensive step.
Common causes are:

- transposed image or exposure axes;
- unequal image, mask, or variance shapes;
- NaN or infinite values;
- nonpositive variance on fitted pixels;
- inverted boolean-mask polarity;
- missing celestial WCS metadata;
- an even kernel or stamp size; or
- labels/splits whose length differs from the sample count.

Compare the input with [Data and artifact contracts](data-artifacts.md).

## A run directory already exists

Many commands preserve completed runs by requiring a fresh output directory.
Choose a new run name or output root. Before removing an old run, confirm that
it contains generated artifacts and that any results you need are backed up.

## Bokeh output is unavailable

Install the visualization profile:

```bash
uv sync --locked --extra viz
```

Numeric workflows run independently of Bokeh. Generate or rebuild the HTML
view after the numeric run succeeds.

## Results differ across devices

First compare shapes, dtypes, finite values, masks, and effective configuration.
Then use a workflow-specific tolerance and a synchronized benchmark. GPU
asynchrony, mixed precision, TF32, reduction order, compilation warmup, and
different mask preprocessing can all change either numbers or timings.

## Reporting a reproducible problem

Follow [Support](../SUPPORT.md). Include the exact command, minimal public or
synthetic input, commit, uv profile, backend/device from `summary.json`, and the
smallest relevant traceback. Use public or synthetic examples and sanitize
credentials, restricted data, and private infrastructure details before sharing.
