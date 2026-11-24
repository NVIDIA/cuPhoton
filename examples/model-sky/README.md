<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Model Sky

Follow one star from a satellite-contaminated image through a motion
measurement and a simulated planetary transit. The notebook shows cuPhoton
reprojection (xRep), image matching and subtraction (xPois), and source
measurement (xFit), with interactive figures and explanations of the science.

Every input is generated from declared models. The example downloads no
astronomical images or catalogs. Known simulation answers remain separate
from the measurement pipeline until the notebook's final reveal. See
[sources.md](sources.md) for the physical assumptions and references.

## Run the notebook

Use a Linux CUDA 13 cuPhoton environment with xRep, xPois and xFit available.
Install the example's additional dependencies, then generate the inputs:

```sh
python -m pip install -r examples/model-sky/requirements.txt
cd examples/model-sky
python model_sky.py --output data
jupyter lab model-sky.ipynb
```

These optional example dependencies include `batman-package` (GPLv3) and
Bokeh, ipykernel, JupyterLab, jupyter-bokeh and nbclient (BSD 3-Clause).
scikit-image uses BSD 3-Clause with additional bundled per-file notices.
No third-party source code is vendored in this example.

Run the cells in order. The source contains 20 code cells without saved
outputs. The short-baseline and blend investigations are initially opt-in.
The inputs occupy approximately 261 MB; generated data and results are
ignored by Git. Use a fresh data directory for preparation.

`CUPHOTON_MODEL_SKY_DATA` and `CUPHOTON_MODEL_SKY_OUTPUT` override the default
`data/` and `results/` directories, relative to the notebook working directory.

## Execute a complete reference

From the repository root, using the GPU environment:

```sh
python examples/model-sky/run.py --output /path/to/new/model-sky-reference
```

The destination must be absent or empty. This copies the example sources,
generates the inputs, and executes a fresh kernel with both investigations
enabled. It writes `model-sky.executed.ipynb`, the measurement results, and
an `execution.json` receipt containing hashes and environment information.
The source notebook and its opt-in defaults stay unchanged.

To run the example's tests, install pytest in the same GPU development
environment and run this command from the repository root:

```sh
python -m pytest examples/model-sky/tests
```

## Preview inside the user guide

Use Quarto and the CPU documentation environment described in
[the documentation instructions](../../docs/README.md). The preview renderer
copies the existing guide and adds Model Sky to its navigation and homepage:

```sh
python examples/model-sky/preview.py \
  --executed /path/to/model-sky-reference/model-sky.executed.ipynb \
  --output /path/to/new/model-sky-guide \
  --site-url http://localhost:8796/
python -m http.server 8796 --bind 0.0.0.0 \
  --directory /path/to/model-sky-guide/guide/_book
```

Open `http://localhost:8796/model-sky.html`. The page retains cell outputs,
terminology hovers, expandable explanations, equations and all six Bokeh
figures. Plot controls run in the browser. Quarto supplies navigation,
search and the page theme; a small adapter connects plot themes and glossary
interactions to the page.

The preview executes the guide's CPU examples and reuses the completed GPU
notebook outputs. It includes downloads of the unexecuted source notebook
and example files, plus a visible link to the notebook in NVIDIA/cuPhoton.
That link uses `main` by default; pass `--source-ref REF` to link to a branch,
tag or commit when reviewing unpublished changes. Rendering and serving do
not publish anything.

The full executed notebook, generated images, data and rendered HTML belong
in build or release assets. Keep the notebook committed to Git output-free.

See [Publishing the Model Sky HTML](PUBLISHING.md) for export commands,
downloadable bundles and the versioned guide publication procedure.
