# Command-line index

cuPhoton installs one executable, `cuphoton`. The same interface is available
from a checkout as `uv run python -m cuphoton`.

```bash
uv run cuphoton --help
uv run cuphoton --version
uv run cuphoton xfit --help
uv run cuphoton xpois --help
uv run cuphoton xpois help fit-kernel
```

The fixed command groups are `xdr`, `xfit`, `xpois`, `xscan`, `xrep`, and
`xray`.
Component-level executables and component-level
`python -m` entry points are not provided.

## xDataReader: `cuphoton xdr`

`benchmark-fits` runs the GPU-native FITS loading benchmark for individual
files or directory scans:

```bash
uv run cuphoton xdr benchmark-fits --help
```

See [xDataReader](components/xdr.md).

## xFit: `cuphoton xfit`

`data-inspect` and `data-validate` check pickle-free NPZ dipole batches whose
arrays are numeric or Unicode;
`fit-dipoles` fits sampled-stamp or analytic Gaussian models and writes
portable fit and uncertainty artifacts. See [xFit](components/xfit.md).

## XPOIS: `cuphoton xpois`

`data-inspect`, `fit-kernel`, `subtract`, `benchmark-backends`,
`evaluate-subtraction`, and `review-bokeh` cover local data inspection,
subtraction, numerical comparison, and review. See
[XPOIS](components/xpois.md).

## XScan: `cuphoton xscan`

XScan has command families for dataset building and validation, pair or triplet
training, inference and evaluation, review queues and annotations, and
controlled reproduction studies. `data-build-xfit-features` creates the
candidate-keyed scalar sidecar used by optional xFit late fusion;
`data-export-xfit-input` creates its dtype-preserving xFit input. Fused
inference and evaluation require the explicit `--use-xfit-features` switch
and a separate `--xfit-feature-dir` location. Evaluation rejects a material
evaluated/validation-split `fit_present` coverage mismatch unless the narrow
`--allow-xfit-coverage-mismatch` calibration override is selected. The
standalone raw-comparison and
Alard--Lupton review servers are available as `review-raw-compare` (`rrc`) and
`review-alard-lupton` (`ral`). Use `cuphoton xscan --help`, then `cuphoton
xscan help <command>` for command-specific contracts. See
[XScan](components/xscan.md).

## xRep: `cuphoton xrep`

`inspect-image`, `reproject-image`, `reproject-stack`, `compare-backends`,
`benchmark-reproject-image`, and `benchmark-backend-variants` cover WCS
inspection, reprojection, parity, and performance. See
[xRep](components/xrep.md).

## XRay: `cuphoton xray`

XRay includes `doctor` and `gpu-policy`; HDF5 probing and trace extraction;
linear-prediction correctness and performance commands; detector artifact,
normalization, comparison, distributed, and merge commands; and report or
visualization commands. The complete list is available from:

```bash
uv run cuphoton xray --help
uv run cuphoton xray help detector-artifact-distributed
```

See [XRay](xray/README.md).

## Configuration and output roots

All groups share command discovery, invariant validation, logging, help,
version plumbing, and XDG path behavior through `cuphoton.core.cli`. Core also
owns the parser backends needed to preserve each command family's established
help and error behavior; component packages only declare invariants.
Workflow-specific YAML `--config` options remain where they describe
scientific work; there is no shared INI configuration option.

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
