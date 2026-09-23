# Architecture

cuPhoton is one Python distribution with independent science components under
one import namespace. A small common layer lets institutions adapt data
adapters and workflows independently.

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

Applications compose these stages with their own adapters, masks, candidate
selection, and execution choices. You can enter at any stage with suitable
local data or arrays. XScan pair models use search and template stamps;
triplet models add a difference image. xFit measurements are an optional
input for models configured to use them.

XRay follows a separate path: delay-indexed detector images become normalized
traces for selected regions, then fitted oscillations and detector maps. Its
inputs and experimental axes differ from optical exposures. Detector-wide
artifact generation requires a GPU and fits row traces within tiles,
repeating each result across that tile's columns. Interpret the maps at that
granularity; CPU paths support individual trace and reference workflows.

## What an exposure contains

An optical exposure includes an image and the information needed to interpret
it: pixel units, variance, bad-pixel flags, a coordinate mapping, a blur model,
and acquisition metadata. This pseudocode summarizes those concepts; each
component defines its concrete input schema:

```text
Exposure {
    image[y, x]       // measured brightness, with explicit units
    variance[y, x]    // uncertainty squared at each pixel
    mask[y, x]        // instrument flags, such as saturation or bad pixels
    wcs               // pixel-to-sky coordinate mapping
    psf               // image pattern produced by one point source
    exposure_time, filter, gain, calibration, provenance // caller metadata
}
```

The pixel and reprojection APIs read pixels and WCS information. Exposure
time, filter, gain, calibration, and exposure provenance remain caller-managed
metadata.

FITS files can hold images and metadata in separate header/data units (HDUs).
An image HDU and a variance HDU may have the same shape but different meanings.
Keep their identities and metadata with the arrays you pass to a workflow.

The World Coordinate System (WCS) describes where pixels point on the sky.
The point-spread function (PSF) describes how light from one point spreads
across nearby pixels. Two images can be aligned yet have different blur:
xRep handles the coordinate mapping, while XPOIS fits the matching filter.
A variance plane describes uncertainty at each pixel. Resampling can also
correlate neighboring pixels' errors; modeling those relationships requires
covariance information.

Instrument masks commonly store flags as bits. cuPhoton's boolean fit masks
use `True` for included pixels. Adapters translate the instrument's
flags into an explicit selection policy. See [Data and artifact
contracts](data-artifacts.md) for each component's shapes, mask rules, and
metadata requirements.

## Package boundaries

`cuphoton.core` owns shared CLI mechanics, application paths, logging, and
invariant evaluation. The six science namespaces own their domain models,
algorithms, adapters, workflow configuration, and workflows. They share
Core's public CLI facade.

xDataReader, xFit, XPOIS, and xRep expose a curated Python surface
for embedding numerical operations. XScan and XRay are primarily
CLI-first, with internal modules available as extension points for workflow
authors.

The portable xFit-to-XScan workflow uses an artifact boundary. XScan
validates a difference-mode xFit run, joins candidates by `candidate_id`,
verifies each fit against the exact difference-stamp hash, and writes a
versioned numeric feature bundle before training or inference. This keeps
the classifier coupled to a feature contract and stable candidate IDs. XScan
can export a pickle-free input archive of numeric and Unicode arrays for
xFit, revalidates stamp hashes whenever a feature bundle is loaded, and pins
the bundle identity in fusion checkpoints.

## Data movement and execution

File loading, numerical computation, and worker coordination are distinct
parts of a run:

- At the file boundary, xDataReader reads and decodes supported FITS images
  into GPU arrays. Storage and driver support determine the transfer path.
  Check the I/O configuration when measuring native GPUDirect Storage.
- At the array boundary, reprojection, convolution, fitting, and inference
  consume different shapes and metadata. Each API's input and return types
  define where its arrays reside and which transfers an application needs.
- At the worker boundary, selected workflows distribute work across processes
  or GPUs. Image-pair batching assigns independent pairs to workers.
  Reusing device memory between stages requires compatible array interfaces
  and explicit management of buffer lifetimes.

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

Select cuTile explicitly after checking compiler/runtime compatibility for
the environment. Use the workflow summary to inspect its resolved execution
settings. Record the backend, device, dtype, and relevant hardware when
comparing runs.

## Configuration and artifacts

Commands accept either explicit options or a YAML configuration, depending
on the workflow. Replace paths in example YAML files with your local data
paths. A persisted run normally contains an effective configuration,
`summary.json`, and an `artifacts/` directory. Some components add traces,
evaluations, checkpoints, or standalone HTML review files.

Structured numeric and JSON artifacts are the reproducible interface. HTML
views and contact sheets are review aids derived from the numeric run.

## Extension points

Prefer a narrow adapter at the component boundary:

- translate local files into the documented array or table contract;
- keep observatory clients and credentials outside cuPhoton;
- pass explicit paths and metadata into the workflow;
- preserve semantic axes and masks explicitly;
- add a synthetic fixture and validation criterion for new behavior.

See [Adapting the workflows](adapting-workflows.md) for a concrete process.
