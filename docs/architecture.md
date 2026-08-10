# Architecture

cuPhoton is one Python distribution with independent science components under
one import namespace. The common layer is deliberately small so institutions
can replace a data adapter or workflow without adopting an application
framework.

```text
installed CLI or Python call
          |
          v
cuphoton.core  -- command discovery, context, logging, invariants
          |
          v
component workflow -- validation, orchestration, artifact contract
          |
          v
CPU / PyTorch / CuPy / Numba-CUDA / cuTile implementation
          |
          v
summary.json + numeric artifacts + optional review output
```

## Package boundaries

`cuphoton.core` owns shared CLI mechanics, application paths, logging, and
invariant evaluation. It does not contain astronomy algorithms or a
process-wide configuration loader. The six science namespaces own their
domain models, algorithms, adapters, workflow configuration, and workflows.
They import Core's public CLI facade instead of carrying private framework
copies.

xDataReader, xFit, XPOIS, and xRep expose a curated Python surface
for embedding numerical operations. XScan and XRay are primarily
CLI-first: their internal modules are available to workflow authors, but are
not a broad compatibility promise.

The xFit-to-XScan boundary is artifact based. XScan validates a portable
difference-mode xFit run, joins candidates by `candidate_id`, verifies each
fit against the exact difference-stamp hash, and writes a versioned numeric
feature bundle before training or inference. The classifier never imports an
xFit solver implementation or assumes fit-table row order. XScan can export a
pickle-free input archive of numeric and Unicode arrays for xFit, revalidates
stamp hashes whenever a feature bundle is loaded, and pins the bundle identity
in fusion checkpoints.

## Execution policy

Where a component accepts `auto`, it selects the most capable installed
backend in its documented order and records the resolution. The intended
orders are:

| Component | Automatic order |
| --- | --- |
| xDataReader | FITS: KvikIO, nvCOMP, and CuPy on CUDA 13; HDF5: Legate |
| xFit | CuPy, then NumPy |
| XPOIS | CuPy, Numba-CUDA, then CPU |
| XScan | PyTorch CUDA, then PyTorch CPU |
| xRep | CuPy, PyTorch CUDA, then CPU |
| XRay | CuPy, then NumPy for supported operations |

cuTile remains explicit because compiler/runtime compatibility must be checked
for a particular environment. A fallback is reported, not silent: workflow
summaries should identify the resolved backend, device, dtype, and relevant
hardware.

## Configuration and artifacts

Commands accept either explicit options or a YAML configuration, depending on
the workflow. Paths in example YAML files are placeholders and should be
overridden for local data. A persisted run normally contains an effective
configuration, `summary.json`, and an `artifacts/` directory. Some components
add traces, evaluations, checkpoints, or standalone HTML review files.

Structured numeric and JSON artifacts are the reproducible interface. HTML
views and contact sheets are derived review aids and should be rebuildable from
the numeric run.

## Extension points

Prefer a narrow adapter at the component boundary:

- translate local files into the documented array or table contract;
- keep observatory clients and credentials outside cuPhoton;
- pass explicit paths and metadata into the workflow;
- preserve semantic axes and masks rather than relying on filename meaning;
- add a synthetic fixture and validation criterion for new behavior.

See [Adapting the workflows](adapting-workflows.md) for a concrete process.
