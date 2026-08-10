# XRay validation visualization

`validation-viz`, `workflow-viz`, and `phonon-viz` create standalone Bokeh HTML
files from persisted numeric products. Install the visualization profile:

```bash
uv sync --locked --extra viz
```

The HTML files are review aids. Keep the trace NPZ, workflow bundle, detector
arrays, and command metadata needed to regenerate them.

## Publishable metadata

Generated HTML uses source basenames and sanitized labels rather than caller
local absolute paths. Workflow bundle manifest version 3 follows the same
rule: it records input file labels, trace filenames, source kind, ROI, fit
settings, array dimensions, thresholds, and counts without persisting the
original HDF5, trace, or detector-artifact directory.

The input artifacts are not rewritten. A trace NPZ supplied by the caller may
still contain its own metadata, and caller-provided titles and file basenames
remain visible. Review those explicit labels before publishing. CLI status
output can include the requested output destination, but that destination is
not embedded in the standalone HTML.

## Trace review

```bash
uv run cuphoton xray validation-viz \
  --trace-dir /path/to/traces \
  --output /path/to/validation.html \
  --title "Validation review"
```

The view can include CPU linear-prediction overlays. Use `--no-fit` to render
only trace and profile data, and `--max-traces` to bound a large directory.

## Workflow bundles

Build a portable numeric bundle while rendering a view:

```bash
uv run cuphoton xray workflow-viz \
  --trace-dir /path/to/traces \
  --bundle-output /path/to/workflow-bundle.npz \
  --output /path/to/workflow.html
```

Render that bundle later without the source trace directory:

```bash
uv run cuphoton xray workflow-viz \
  --bundle /path/to/workflow-bundle.npz \
  --output /path/to/workflow.html
```

`workflow-viz` can also combine trace, HDF5 ROI, and detector-artifact context;
use `cuphoton xray help workflow-viz` for the mutually optional input modes.

## Phonon-style view

```bash
uv run cuphoton xray phonon-viz \
  --workflow-bundle /path/to/workflow-bundle.npz \
  --output /path/to/phonon.html
```

Detector-artifact mode accepts a detector artifact directory and optional
x/y bounds. Record the amplitude threshold and fit parameters with any
published view.
