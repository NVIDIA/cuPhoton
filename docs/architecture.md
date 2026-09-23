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

## How the science components fit together

The optical components cover different operations on images. A workflow that
looks for changes between observations might connect them as follows:

```text
local FITS images
    |
    v
xDataReader: decode and load pixels onto the GPU
    |
    v
xRep: resample onto a common sky grid, when needed
    |
    v
XPOIS: match blur and background, then subtract
    |
    v
candidate detection and stamp extraction (caller or dataset workflow)
    |
    +--> xFit: measure dipole shapes (optional) --+
    |                                          |
    v                                          v
XScan: score candidate images, optionally with xFit measurements
    |
    v
scores, evaluation, and human review
```

This describes data relationships. Applications still supply adapters,
masks, candidate selection, and execution choices; it is not a single
end-to-end command. You can enter at any stage with suitable local data or
arrays. XScan pair models can use search and template stamps without a
difference image, and image-only models do not require xFit.

XRay follows a separate path: delay-indexed detector images become normalized
traces for selected regions, then fitted oscillations and detector maps. Its
inputs and experimental axes differ from optical exposures. Detector-wide
artifact generation requires a GPU and fits row traces within tiles,
repeating each result across that tile's columns. Interpret the maps at that
granularity; CPU paths support individual trace and reference workflows.

## An exposure is more than pixels

An optical exposure includes an image and the information needed to interpret
it: pixel units, variance, bad-pixel flags, a coordinate mapping, a blur model,
and acquisition metadata. The following is a conceptual data structure, not
a cuPhoton class or a required file schema:

```text
Exposure {
    image[y, x]       // measured brightness, with explicit units
    variance[y, x]    // uncertainty squared at each pixel
    mask[y, x]        // instrument flags, such as saturation or bad pixels
    wcs               // pixel-to-sky coordinate mapping
    psf               // image pattern produced by one point source
    time, filter, calibration, provenance
}
```

FITS files can hold images and metadata in separate header/data units (HDUs).
An image HDU and a variance HDU may have the same shape but different meanings.
Keep their identities and metadata with the arrays you pass to a workflow.

The World Coordinate System (WCS) describes where pixels point on the sky.
The point-spread function (PSF) describes how light from one point spreads
across nearby pixels. Two images can be aligned yet have different blur:
xRep handles the coordinate mapping, while XPOIS fits the matching filter.
Resampling can also correlate neighboring pixels' errors; a variance plane
alone does not describe those correlations.

Instrument masks commonly store flags as bits. cuPhoton's boolean fit masks
use `True` for included pixels, so adapters must translate the instrument's
flags into an explicit selection policy. See [Data and artifact
contracts](data-artifacts.md) for each component's shapes, mask rules, and
metadata requirements.

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

The portable xFit-to-XScan workflow uses an artifact boundary. XScan
validates a difference-mode xFit run, joins candidates by `candidate_id`,
verifies each fit against the exact difference-stamp hash, and writes a
versioned numeric feature bundle before training or inference. The classifier never imports an
xFit solver implementation or assumes fit-table row order. XScan can export a
pickle-free input archive of numeric and Unicode arrays for xFit, revalidates
stamp hashes whenever a feature bundle is loaded, and pins the bundle identity
in fusion checkpoints.

## Data movement and execution

File loading, numerical computation, and worker coordination are distinct
parts of a run:

- At the file boundary, xDataReader reads and decodes supported FITS images
  into GPU arrays. Native GPUDirect Storage depends on the actual I/O path;
  a device array alone is not evidence of a direct storage transfer.
- At the array boundary, reprojection, convolution, fitting, and inference
  consume different shapes and metadata. Check each API's input and return
  types before assuming intermediate results stay on the GPU.
- At the worker boundary, selected workflows distribute work across processes
  or GPUs. A batch of independent image pairs is different from dividing one
  image solve across GPUs; worker reuse alone does not imply device-memory
  reuse between stages.

## Execution policy

Where a component accepts `auto`, it selects the most capable installed
backend in its documented order and records the resolution. The intended
orders are:

| Component | Automatic order |
| --- | --- |
| xDataReader | KvikIO, nvCOMP, and CuPy on CUDA 13 |
| xFit | CuPy, then NumPy |
| XPOIS | Constant: CuPy, Numba-CUDA, then CPU; spatial ALS: CuPy, then CPU |
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
