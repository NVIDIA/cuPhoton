# Troubleshooting

## Confirm the environment first

Run commands through the environment created for this checkout:

```bash
uv run --no-sync python - <<'PY'
import sys
import cuphoton
print(sys.executable, cuphoton.__version__, cuphoton.__file__)
PY
uv lock --check
uv run --no-sync cuphoton xray doctor
```

If an executable is missing, rerun `uv sync --locked` with every extra needed
by your selected profile. If the import resolves to another checkout, inspect
`sys.executable`, `cuphoton.__file__`, and any `PYTHONPATH` entries before
debugging the workflow.

## A GPU run used CPU

Workflows with an `auto` backend may select CPU. Check the workflow or
quickstart summary for the resolved backend and device. Then verify that:

- the environment was synced with `--extra gpu`;
- the NVIDIA driver is visible through `nvidia-smi`;
- PyTorch reports `torch.cuda.is_available()`;
- CuPy can allocate and synchronize a small array; and
- `CUDA_VISIBLE_DEVICES` includes the intended GPU, if set.

Use `--require-gpu` in the synthetic runner to require a CUDA run. Select the
`cutile` backend explicitly with a compatible Python, `cuda-tile` runtime,
and TileIR compiler.

## CUDA package or driver mismatch

cuPhoton supports CUDA 13. If an environment mixes CUDA 12 and CUDA 13
packages, create a fresh environment with the locked GPU profile from
[Getting started](getting-started.md#clone-and-select-a-profile). The lock file
selects CUDA 13 dependencies; the host also needs a compatible NVIDIA driver.
Record the driver, GPU, Python, and resolved package versions in bug reports.

## FITS reading used Astropy or could not load xDR

The FITS reader and numerical backend are separate choices. A GPU fit can
read pixels through Astropy and transfer them to the GPU. Inspect the run's
FITS I/O metadata for `requested_reader`, `reader`, and `fallback_reason`.
With `auto`, unsupported scaling, null values, compression, or uncompressed
image sections select Astropy. Explicit `--fits-reader xdr` rejects unsupported
inputs or unavailable native/GPU dependencies.

Source and editable installations do not compile xDR by default, even with
the `io` or `gpu` extra. Check native loading in the prepared environment:

```bash
uv run --no-sync python -c \
  'from cuphoton.xdr.nvcomp_batch import cpp_helper_available; print(cpp_helper_available())'
```

For a checkout, follow the [native build instructions](components/xdr.md#native-extension-availability).
For a release wheel, install its `io` extra and confirm that the imported
package comes from that wheel. `xray doctor` checks xRay dependencies; it
does not check the xDR extension.

Once a FITS payload read starts, errors propagate instead of retrying through
another reader. Check file accessibility, the selected HDU, and the native
error. GPUDirect Storage also requires host and storage configuration; use
`KVIKIO_COMPAT_MODE=ON` for ordinary file I/O when GDS is not configured.

### An explicit compression choice is unavailable

`--xdr-gzip-decoder gzip` requires native Gzip support;
`--xdr-decompression-backend cuda` requires an extension with backend selection.
An importable extension may predate those entry points. After updating a source
checkout, rebuild it with the [native build instructions](components/xdr.md#native-extension-availability)
in the environment running your command. Use `auto` when capability fallback
is acceptable. A Python prefetcher still performs GPU decoding, and an `auto`
backend in a report does not prove that a hardware decompression engine ran.
See [runtime choices](components/xdr.md#runtime-choices) for the separate
reader, decoder and backend policies.

### Confirm the decompression engine

Enable [nvCOMP logging](https://docs.nvidia.com/cuda/nvcomp/getting_started.html#logging)
for a diagnostic run. Level 5 includes API calls as well as informational
messages; use a separate log file to keep them out of the command output:

```bash
NVCOMP_LOG_LEVEL=5 NVCOMP_LOG_FILE=nvcomp.log \
  cuphoton xdr benchmark-fits --hdu-indices 1 \
  --xdr-gzip-decoder gzip --xdr-decompression-backend auto exposure.fits
```

In nvCOMP 5.2, `Trying HW decompression` records an attempt, while
`Launching HW decompression` and `HW Decompression successful` identify
hardware execution. `launching SM decompression` identifies the CUDA path.
Inspect the messages for the measured calls; a hardware attempt alone does
not prove success. Compare with `--xdr-decompression-backend cuda`, and disable
logging for timings. See [hardware eligibility](components/xdr.md#hardware-eligibility-and-diagnostics)
for buffer and tile-size requirements. FITS receipts retain requested choices,
including choices that did not apply when the selected reader was Astropy.

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

Add `viz` to the profile you use. For the CPU development profile:

```bash
uv sync --locked --extra dev --extra torch --extra photometry --extra viz
```

For GPU development, use `--extra dev --extra gpu --extra viz` instead.
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
