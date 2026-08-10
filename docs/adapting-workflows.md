# Adapting the workflows

cuPhoton is meant to be changed. The safest adaptation keeps the numerical
workflow and artifact contract visible while isolating institution-specific
data discovery.

## 1. Establish a baseline

Run the synthetic CPU quickstart and save its `summary.json` outside the
repository:

```bash
uv run python examples/run_quickstarts.py \
  --profile cpu \
  --output-dir /tmp/cuphoton-baseline
```

If the target system has CUDA, run the same component with `--require-gpu` and
compare shapes, finite-value checks, scientific metrics, and timings.

## 2. Map the local data contract

Document each source field, unit, axis, mask convention, and coordinate frame.
Translate it to the component contract in
[Data and artifact contracts](data-artifacts.md). Do not infer semantics from
directory names when metadata can be carried explicitly.

Keep remote clients, authentication, and site catalogs in a separate staging
step. The cuPhoton invocation should receive local paths or arrays that another
researcher could reproduce independently.

## 3. Add the smallest adapter

Prefer a converter or loader at the component edge over changes to the core
solver. Validate before allocating large device arrays:

- rank, shape, and axis order;
- dtype and byte order;
- finite values and variance positivity;
- mask polarity and bit-plane meaning;
- WCS availability and units;
- sample, exposure, or detector identifiers.

If several components need the same generic command, invariant, logging, or
application-context behavior, place it in `cuphoton.core`. Workflow-specific
configuration and domain data models stay with their component.

## 4. Preserve provenance

Record the effective configuration, code revision, input identity, random
seed, selected backend/device, dtype, GPU and driver when applicable, output
shapes, and validation metrics. Prefer relative paths inside a run so the run
directory can be moved. Never write credentials or private service URLs into
tracked examples or reusable reports.

## 5. Validate numerically and visually

Choose tolerances before comparing implementations. Check both low-level array
agreement and workflow-level outcomes such as objective value, PSF
normalization, residual distribution, classification metrics, or WCS alignment.
Use the optional Bokeh views to inspect failures, but base acceptance on the
persisted numeric artifacts.

For performance claims, use warmups, multiple synchronized repetitions, the
same input and dtype, and the same GPU. Report the median and peak memory; do
not compare an end-to-end path with a kernel-only path.

## 6. Contribute the reusable part

A contribution should include the public contract, a synthetic or freely
redistributable fixture, tests, documentation, and dependency impact. Keep
local catalog logic, data locations, run outputs, and unreviewed notebooks out
of the repository. See [Contributing](../CONTRIBUTING.md).
