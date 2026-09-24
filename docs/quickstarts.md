# Synthetic quickstarts

The checkout quickstart runner creates small deterministic inputs locally and
exercises the science components. Use it to smoke-test the environment and
explore the artifact formats. Scientific benchmarks use representative data
and a measurement protocol suited to the workflow.

## Run all components

The default profile prefers GPU backends and falls back to complete CPU paths
when CUDA is unavailable:

```bash
uv sync --locked --extra dev --extra gpu --extra viz
uv run python examples/run_quickstarts.py
```

For a deterministic CPU run:

```bash
uv sync --locked --extra dev --extra torch --extra viz
uv run python examples/run_quickstarts.py --profile cpu
```

To require CUDA:

```bash
uv run python examples/run_quickstarts.py --require-gpu
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
| XPOIS | two matched 2D source images | kernel fit, matched image, and residual |
| XScan | labeled image stamps and fixed splits | one-epoch training and evaluation smoke |
| xRep | a small FITS image with a valid celestial WCS | reprojection and mask propagation |
| XRay | deterministic detector traces | extraction or linear-prediction analysis |

## Output contract

The output root contains a subdirectory for each selected component and a root
`summary.json`. The same summary is printed to standard output. It records:

- the requested profile;
- the resolved backend, device, and dtype for each component;
- relevant hardware information;
- input seeds and shapes; and
- paths to the generated component artifacts.

Delete the output directory before rerunning with the same destination. Keep
quickstart outputs outside version control; regenerate them from the command
and recorded seed.

## Interpreting fallback

An `auto` run is successful on CPU or GPU. Use `summary.json` to confirm the
backend used. A performance or CUDA correctness check should use
`--require-gpu`, record the GPU and driver, and synchronize device work in the
benchmark harness.
