# Command-line index

cuPhoton provides one Python console entry point, `cuphoton`. The same
interface is available from a checkout as `uv run python -m cuphoton`.
Installations also include the low-level `cuphoton-openmpi-rank-exec` helper
used to bind Open MPI ranks before Python starts.

```bash
uv run cuphoton --help
uv run cuphoton --version
uv run cuphoton xfit --help
uv run cuphoton xpois --help
uv run cuphoton xpois help fit-kernel
```

Access each component through the fixed command groups: `xdr`, `xfit`, `xpois`,
`xscan`, `xrep`, and `xray`.

FITS-consuming commands may expose `--fits-reader auto|astropy|xdr` separately
from the compute backend. GPU workflows use `auto` to prefer xDR when the
requested read is supported and the GPU dependencies are available. The
shared reader records why it falls back to Astropy. CPU xFit, xPois, and xRep
workflows resolve `auto` directly to Astropy. Explicit `xdr` requires an
eligible read. The pipeline commands preserve each manifest's reader policy
unless an override is supplied.
Check command help for the applicable default and the
[FITS reader guide](components/xdr.md#read-fits-images-in-a-workflow) for
scaling, compression, and section-read rules.

Applicable FITS commands and `xdr benchmark-fits` also expose
`--xdr-postprocess`, `--xdr-gzip-decoder` and `--xdr-decompression-backend`.
Omitted flags preserve manifest choices; otherwise the xDR runtime defaults
are `auto`. These settings apply when xDR is the selected FITS reader. See
[runtime choices](components/xdr.md#runtime-choices) for values, Python
keywords and capability fallback.

## xDataReader: `cuphoton xdr`

`benchmark-fits` runs the GPU-native FITS loading benchmark for individual
files or directory scans:

```bash
uv run cuphoton xdr benchmark-fits --help
```

See [xDataReader](components/xdr.md).

## xFit: `cuphoton xfit`

`data-inspect` and `data-validate` check dipole inputs: pickle-free NPZ batches
of numeric or Unicode arrays, or JSON FITS candidate manifests. `fit-dipoles`
fits sampled-stamp or analytic Gaussian models and writes portable fit and
uncertainty artifacts. FITS manifests select image HDUs and candidate regions;
`--fits-reader` controls loading independently of `--backend`.
`fit-dipoles --executor dragon|mpi` distributes candidate chunks; local
execution remains the default. `--warmup-rounds` or `--measure-rounds` opts
into persistent workers and a separate artifact directory for each round.
See [xFit](components/xfit.md).

## xPois: `cuphoton xpois`

`data-inspect`, `fit-kernel`, `subtract`, `fit-batch`, `benchmark-backends`,
`evaluate-subtraction`, and `review-bokeh` cover local data inspection,
subtraction, distributed whole-pair execution, numerical comparison, and
review. For `fit-batch`, `--executor mpi|dragon` selects process orchestration
while `--backend cupy|numba-cuda|cutile` selects the numerical implementation
inside each worker. Both are explicit; launcher and Dragon transport options
remain outside cuPhoton. See [xPois](components/xpois.md).

Select the spatially varying alternating-linear-least-squares model and its
single-GPU CuPy backend explicitly:

```bash
uv run cuphoton xpois fit-kernel \
  --reference /path/to/reference.fits \
  --target /path/to/target.fits \
  --solver spatial-als \
  --backend cupy

uv run cuphoton xpois benchmark-backends \
  --reference /path/to/reference.fits \
  --target /path/to/target.fits \
  --solver spatial-als \
  --backends cpu,cupy \
  --reference-backend cpu
```

The Dragon batch executor accepts the same spatial solver configuration and
assigns each image-pair fit to one explicitly placed GPU worker:

```bash
.venv/bin/dragon examples/xpois/dragon_batch.py \
  --manifest /shared/manifests/fixed-32.yaml \
  --solver spatial-als \
  --backend cupy \
  --max-workers 4
```

For single-image spatial ALS commands, `--backend auto` prefers CuPy when a
usable CUDA device is available and otherwise uses the CPU reference
implementation. Explicit `--backend cupy` requires a usable CUDA device and
fails if one is unavailable. The CPU and CuPy paths fit the same FP64 model
from the supplied image pair. The batch command requires `--backend cupy`
for spatial ALS and rejects `auto`. One solver configuration applies to every
pair in the batch; each complete spatial solve runs on one GPU.

## xScan: `cuphoton xscan`

xScan has command families for dataset building and validation, pair or
triplet training, inference and evaluation, review queues and annotations,
and controlled reproduction studies. `data-build-xfit-features` creates the
candidate-keyed scalar sidecar used by optional xFit late fusion;
`data-export-xfit-input` creates its dtype-preserving xFit input. Fused
inference and evaluation require the explicit `--use-xfit-features` switch
and a separate `--xfit-feature-dir` location. Evaluation rejects a material
evaluated/validation-split `fit_present` coverage mismatch unless the narrow
`--allow-xfit-coverage-mismatch` calibration override is selected. The
standalone raw-comparison and Alard--Lupton review servers are available as
`review-raw-compare` (`rrc`) and `review-alard-lupton` (`ral`). Use
`cuphoton xscan --help`, then `cuphoton xscan help <command>` for
command-specific contracts. See [xScan](components/xscan.md).

`infer-real-bogus --executor dragon|mpi` distributes complete minibatches;
`run-pipeline --executor dragon|mpi|processes|threads` distributes complete
image pairs through xPois, xFit and xScan. Optional round flags retain the
workers across passes. Dragon and MPI require a shared input/output
filesystem and the matching runtime launcher. The local `processes` and
`threads` pipeline executors share one visible GPU; see
[GPU sharing](components/xscan.md#share-a-gpu-between-image-pairs).

`infer-real-bogus` consumes packaged datasets. Direct FITS candidate scoring
uses the CUDA Python API `cuphoton.xscan.fits_inference.predict_fits`.
`run-pipeline` accepts NPY or FITS image descriptors in its manifest and
`benchmark-pipeline` compares the persistent pipeline with separate stages.

## xRep: `cuphoton xrep`

`inspect-image`, `reproject-image`, `reproject-stack`, `compare-backends`,
`benchmark-reproject-image`, and `benchmark-backend-variants` cover WCS
inspection, reprojection, parity, and performance. See
[xRep](components/xrep.md).

## xRay: `cuphoton xray`

xRay includes `doctor` and `gpu-policy`; HDF5 probing and trace extraction;
linear-prediction correctness and performance commands; detector artifact,
normalization, comparison, distributed, and merge commands; and report or
visualization commands. The complete list is available from:

```bash
uv run cuphoton xray --help
uv run cuphoton xray help detector-artifact-distributed
```

See [xRay](xray/README.md).

## Configuration and output roots

All groups share command discovery, invariant validation, logging, help,
version plumbing, and XDG path behavior through `cuphoton.core.cli`. Core also
owns the parser backends needed to preserve each command family's established
help and error behavior; component packages declare invariants.
Workflow-specific YAML `--config` options configure the scientific work within
each component.

Component configuration, state, and data live under a common product root:

```text
$XDG_CONFIG_HOME/cuphoton/<group>
$XDG_STATE_HOME/cuphoton/<group>
$XDG_DATA_HOME/cuphoton/<group>
```

Runs are written under the group's state `runs` directory and logs under its
state `logs` directory unless a command accepts and receives an explicit
output path. Capture standard output when a command emits JSON, and inspect
the persisted `summary.json` before relying on the backend or device used.
