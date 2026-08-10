# Troubleshooting

## Confirm the environment first

Run commands through the environment created for this checkout:

```bash
uv run python -c 'import sys, cuphoton; print(sys.executable, cuphoton.__version__)'
uv lock --check
uv run xray doctor
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
- `CUDA_VISIBLE_DEVICES` has not hidden the intended GPU.

Use `--require-gpu` in the synthetic runner when fallback should fail. The
`cutile` backend is never selected implicitly; it needs a compatible Python,
`cuda-tile` runtime, and TileIR compiler.

## CUDA package or driver mismatch

cuPhoton supports CUDA 13 dependency variants only. Remove mixed CUDA 12/13
packages from the environment and recreate it from `uv.lock`. A system CUDA
toolkit is not a substitute for a sufficiently new driver. Record the driver,
GPU, Python, and resolved package versions in bug reports.

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

Many commands refuse to overwrite a completed run. Choose a new run name or
output root. Remove an old run only after confirming it is a generated artifact
and not the only copy of a result.

## Bokeh output is unavailable

Install the visualization profile:

```bash
uv sync --locked --extra viz
```

Numeric workflow artifacts do not require Bokeh. Generate or rebuild the HTML
view after the numeric run succeeds.

## Results differ across devices

First compare shapes, dtypes, finite values, masks, and effective configuration.
Then use a workflow-specific tolerance and a synchronized benchmark. GPU
asynchrony, mixed precision, TF32, reduction order, compilation warmup, and
different mask preprocessing can all change either numbers or timings.

## Reporting a reproducible problem

Follow [Support](../SUPPORT.md). Include the exact command, minimal public or
synthetic input, commit, uv profile, backend/device from `summary.json`, and the
smallest relevant traceback. Do not include credentials, restricted data, or
private infrastructure details.
