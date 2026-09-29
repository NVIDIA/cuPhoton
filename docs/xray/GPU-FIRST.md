# xRay GPU-first behavior

xRay uses CuPy for high-throughput trace batches and detector-wide analysis.
NumPy paths provide deterministic smoke tests, algorithm inspection, and
correctness references for supported operations.

## Before a run

```bash
uv run cuphoton xray doctor
uv run cuphoton xray gpu-policy
```

Confirm the selected environment, GPU, driver, CUDA runtime, and available
memory. Run a representative ROI before committing to a full detector.

## Fallback and requirements

The synthetic quickstart reports `gpu` or `cpu`; component manifests identify
the concrete CuPy or NumPy implementation where applicable. Read the JSON or
run summary to confirm the backend used. Detector-wide commands that require
a GPU identify the `gpu` extra in their help and error messages.

Use the checkout runner to distinguish a portable smoke from a CUDA gate:

```bash
uv run python examples/run_quickstarts.py --component xray --profile cpu \
  --output-dir /tmp/xray-cpu-quickstart
uv run python examples/run_quickstarts.py --component xray --require-gpu \
  --output-dir /tmp/xray-gpu-quickstart
```

Use the [CUDA environment](ENVIRONMENT.md#cuda-13-development) for the GPU
run and choose new output directories if these already exist.

## Performance records

For publishable measurements, record the command, code revision, Python and
package versions, backend, dtype, GPU, driver, CUDA runtime, input identity and
shape, ROI, warmup, repetition count, synchronization, and output manifest.
Keep source HDF5, detector artifacts, and benchmark logs outside the checkout.
