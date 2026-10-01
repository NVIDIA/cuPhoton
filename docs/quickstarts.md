# Synthetic quickstarts

The checkout quickstart runner creates small deterministic inputs locally and
exercises the science components. Use it to smoke-test the environment and
explore the artifact formats. Scientific benchmarks use representative data
and a measurement protocol suited to the workflow.

## Run all components

The default profile prefers GPU backends and falls back to complete CPU paths
when CUDA is unavailable. On Linux, install the combined GPU dependencies:

```bash
uv sync --locked --extra dev --extra gpu --extra viz
uv run python examples/run_quickstarts.py \
  --output-dir /tmp/cuphoton-auto-quickstart
```

For a deterministic CPU run:

```bash
uv sync --locked --extra dev --extra torch --extra viz
uv run python examples/run_quickstarts.py --profile cpu \
  --output-dir /tmp/cuphoton-cpu-quickstart
```

To require CUDA, use the GPU environment from the first example:

```bash
uv sync --locked --extra dev --extra gpu --extra viz
uv run python examples/run_quickstarts.py --require-gpu \
  --output-dir /tmp/cuphoton-gpu-quickstart
```

Use `--output-dir` to select the destination. The default is
`quickstart-output/`:

```bash
uv run python examples/run_quickstarts.py \
  --profile cpu \
  --output-dir /tmp/cuphoton-quickstart
```

## Run one component

Pass `--component` one or more times:

```bash
uv run python examples/run_quickstarts.py \
  --component xfit \
  --output-dir /tmp/xfit-quickstart
```

Valid component names are `xfit`, `xpois`, `xscan`, `xrep`, and `xray`.

## What each quickstart covers

| Component | Synthetic input | Exercised path |
| --- | --- | --- |
| xFit | analytic Gaussian dipole stamps and perturbed parameters | batched nonlinear fit and uncertainty artifacts |
| xPois | two matched 2D source images | kernel fit, matched image, and residual |
| xScan | labeled image stamps and fixed splits | one-epoch training and evaluation smoke |
| xRep | a small FITS image with a valid celestial WCS | reprojection with FITS/WCS output |
| xRay | a deterministic on/off HDF5 pair | trace extraction and linear-prediction analysis |

xDR is not a selectable quickstart component. Its native FITS extension and
device-transfer workflow have separate prerequisites; see the
[xDR guide](components/xdr.md) and the
[imaging walkthrough](../examples/imaging-pipeline/run_imaging_pipeline.ipynb).

## Output contract

The output root contains a subdirectory for each selected component and a root
`summary.json`. The same summary is printed to standard output. It records:

- the requested profile;
- the resolved backend, device, and dtype for each component;
- relevant hardware information;
- input seeds and shapes; and
- paths to the generated component artifacts.

The runner requires a new output directory. Choose a different destination
for each run, or remove a previous output directory when its artifacts are
no longer needed. Keep quickstart outputs outside version control;
regenerate them from the command and recorded seed.

## Interpreting fallback

An `auto` run is successful on CPU or GPU. Use `summary.json` to confirm the
backend used. A performance or CUDA correctness check should use
`--require-gpu`, record the GPU and driver, and synchronize device work in the
benchmark harness.
