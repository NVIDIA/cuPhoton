# XRay

`cuphoton.xray` provides X-ray trace extraction, linear-prediction analysis,
detector artifact generation, numerical validation, and standalone review
views. It is available through the `cuphoton xray` command group.

XRay is GPU-first for high-throughput detector work. Supported operations fall
back to NumPy for CPU smoke and correctness runs; commands that require a GPU
report that requirement explicitly.

## Install and smoke test

```bash
# CUDA 13
uv sync --locked --extra dev --extra gpu --extra viz
uv run cuphoton xray doctor
uv run python examples/run_quickstarts.py --component xray --require-gpu

# CPU
uv sync --locked --extra dev --extra viz
uv run python examples/run_quickstarts.py --component xray --profile cpu
```

## Command groups

| Goal | Commands |
| --- | --- |
| Environment | `doctor`, `gpu-policy` |
| Input inspection | `data-probe`, `roi-candidates` |
| Trace work | `extract-trace`, `trace-smoke` |
| Linear prediction | `linear-prediction-*`, `prediction-roots-benchmark`, `model-order-sweep` |
| Detector products | `detector-mask`, `detector-artifacts`, `detector-artifact-normalize`, `detector-artifact-compare` |
| Distributed detector work | `detector-artifact-distributed`, `detector-artifact-merge` |
| Review | `report`, `validation-viz`, `workflow-viz`, `phonon-viz` |

List the complete current command surface with:

```bash
uv run cuphoton xray --help
uv run cuphoton xray help extract-trace
```

## Single-node HDF5 workflow

XRay recognizes two on/off cube layouts. Both files in a pair must use the
same schema and array shapes.

| Schema | Image cube | Delay axis | Entry counts | Normalization |
| --- | --- | --- | --- | --- |
| `cropped-cube` | `imgs` `(frame, y, x)` | `scan_var` `(frame,)` | `bin_count` `(frame,)` | `i0` and `i0_ipm3` `(frame,)`; `ROI` is also required |
| `legate-cube` | `jungfrau1M_data` `(frame, y, x)` | `binVar_bins` `(frame,)` | `nEntries` `(frame,)` | `ipm3__sum`/`ipm2__sum`, or `ipm5__sum`/`ipm4__sum`, each `(frame,)` |

Probe first, extract a small row batch, and inspect model-order behavior before
running the detector-wide GPU path:

```bash
uv run cuphoton xray data-probe \
  --h5dir /path/to/input --fon on.h5 --foff off.h5 --json

uv run cuphoton xray extract-trace \
  --h5dir /path/to/input --fon on.h5 --foff off.h5 \
  --roi-lower 0 0 --roi-dim 64 64 --row-y 0 --row-y 16 \
  --output-dir /path/to/traces --output-prefix sample --json

uv run cuphoton xray model-order-sweep \
  --trace-dir /path/to/traces --min-components 2 \
  --max-components 12 --step 2 --json

uv run cuphoton xray detector-artifacts \
  --h5dir /path/to/input --fon on.h5 --foff off.h5 \
  --output-dir /path/to/artifacts --roi-lower 0 0 --roi-dim 64 64 \
  --zero-offset-index 0 --components 6 --json
```

`--zero-offset-index 0` is illustrative; select a physically appropriate fit
start for the input scan. Start with a representative ROI and inspect the
manifest, fit-status array, and numerical outputs before scaling out.

## Input and artifact boundary

Pass HDF5, trace NPZ, detector-array, and output paths explicitly. The package
does not select a data root. Keep source datasets, detector dumps, generated
reports, dashboards, and benchmark logs outside the repository.

Before a long run, use `data-probe` to confirm the HDF5 schema and on/off shape
agreement. Preserve the ROI, excluded rows, normalization fields, fit
parameters, package revision, backend/device, and GPU/driver with the result.

See [Data and artifact contracts](../data-artifacts.md#xray-hdf5-and-trace-products).

## Detailed guides

- [Environment setup](ENVIRONMENT.md)
- [GPU-first behavior](GPU-FIRST.md)
- [Distributed detector artifacts](DISTRIBUTED-DETECTOR-ARTIFACTS.md)
- [Validation visualization](VALIDATION-VIZ.md)
