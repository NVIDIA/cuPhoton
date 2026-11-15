<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Model Scene

What can blurred exposures actually reveal? This notebook reconstructs a
static modeled scene with xImage, calibrates PSFs from separate modeled star
observations, and measures recovery against independent truth. Interactive
figures connect individual exposures with predictions and residuals.

Every input comes from the analytic generator in this directory. The example
uses no survey images, calibration products, catalogs or downloaded fixtures.
The physical renderer integrates continuous Gaussian mixtures over detector
pixels. A separate discrete-operator control quantifies representation error.
A 96×96 fitting region provides context around the central 48×48 science
window. A separate compact-context case demonstrates halo contamination.
The experiment uses a zero-dither grid and makes no alignment qualification.

## Execute

Use a fresh Linux Python 3.12+ environment with CUDA 13 and a compatible NVIDIA
GPU. From a checkout of the exact cuPhoton revision being evaluated:

```sh
python -m pip install -e '.[ximage]'
python -m pip install -r examples/model-scene/requirements.txt
python examples/model-scene/run.py --profile quick --device cuda \
  --gpu-lock /path/shared-by-all-gpu-users/gpu0.lock \
  --output /path/outside/git/new-scene-smoke
```

Pass the same advisory lock path to other cooperating GPU workloads. The runner
retains that lock through the study and the fresh notebook kernel. Inspect GPU
use before launching; a process that ignores the advisory lock is not blocked
by it. The destination must be absent, including absent as a symbolic link.
The runner never overwrites an earlier execution.

The quick profile exercises the bundle contracts, solver APIs, investigations
and plots with two trials and ten scene iterations. It substantiates no
scientific-performance claim. `--device cpu` provides an API smoke route.
The reference profile runs the frozen protocol, including at least 50 paired
science-noise realizations and 50 independent calibration draws:

```sh
python examples/model-scene/run.py --profile reference --device cuda \
  --gpu-lock /path/shared-by-all-gpu-users/gpu0.lock \
  --output /path/outside/git/new-scene-reference
```

`protocol.json` records the measurement definitions and seed ranges. A reference
run refuses an unfrozen protocol or a change to its scientific source hashes.
The reproducible development sweep uses separate seeds and a catalog that shifts the two isolated stars by two pixels:

```sh
python examples/model-scene/develop.py --device cuda \
  --gpu-lock /path/shared-by-all-gpu-users/gpu0.lock \
  --output /path/outside/git/new-scene-development
```

Development evidence determines the iteration budget before
test draws. The Huber threshold is declared in advance, not tuned by this sweep.
An invalidating scientific fix requires a new protocol revision
and fresh test seeds; retain the failed attempt and its receipt.

The source notebook has no outputs, execution counts, attachments or datasets.
`run.py` creates a source copy, native workflow artifacts, numerical study,
execution log, executed notebook and an execution receipt. It records source,
protocol and output hashes, exact source revision and package versions.
`CUPHOTON_MODEL_SCENE_OUTPUT` locates those results for notebook execution.
The notebook preserves warnings and recorded outputs. Work offline after
installing dependencies; no data download is part of generation or execution.

## Interpret the measurements

The primary endpoint is fractional error of radius-three-pixel aperture flux
at a common 2.4-pixel Gaussian resolution. Compare each fixed aperture with
the identical aperture on target truth. Apertures include neighboring light;
they do not measure deblended total source flux. Per-source summaries retain
brightness and crowding labels. This is inference about the fixed modeled
catalog and independent noise draws, rather than a population calibration.

The reference compares L2, Huber and a coadd on paired draws, then investigates
PSF-width errors, finite calibration, outliers, explicit masking, variance
rescaling, photon/read noise and a transient. Held-out frames test forward
predictions. Native coadd-relative metrics are secondary diagnostics. See the
notebook's derivations and caveats before interpreting a residual statistic
or a visually sharp feature.

## Inspect a static export

Use the CPU documentation environment and Quarto described in
[the documentation guide](../../docs/README.md) and
[developer tooling](../../dev/README.md):

```sh
python examples/model-scene/preview.py \
  --executed /path/new-scene-reference/model-scene.executed.ipynb \
  --output /path/outside/git/new-scene-preview \
  --source-ref YOUR_PUSHED_SOURCE_REF \
  --controller-python /path/to/developer-env/bin/python
python -m http.server 8798 --bind 0.0.0.0 \
  --directory /path/new-scene-preview/guide/_book
```

Open `http://localhost:8798/model-scene.html`. The chapter embeds its JavaScript
and numeric data. Each experiment selector chooses saved results. It requires
no Python server callbacks, kernel, WebSocket, API or CDN. Individual standalone
chapter viewing also works with `file://`; guide navigation needs the rest of
the exported directory. Display-range changes retain the original pixel values.

Use `--controller-python` when the developer tooling dependencies (including
Beautiful Soup) live in a separate environment from the CPU guide renderer.

The preview verifies the execution receipt and source hashes, executes the
ordinary CPU guide chapters, preserves the completed GPU notebook outputs,
and invokes the current version postprocessor and snapshot checks. It does
not modify the central publisher. Source links must name a pushed ref.

Run the numerical tests with `python -m pytest examples/model-scene/tests`.
Workflow controls additionally require the xImage dependencies.

[Publishing instructions](PUBLISHING.md) describe packaging and the current
versioned-Pages integration gap. Opening or merging this source PR does not
publish the rendered chapter.

The frozen scientific source was replayed with the development command above.
Its twelve rows agree numerically with the selection sweep. The external
`development-v2-replay/development.json` receipt has SHA-256
`355ce588c2ccc039a1d1bbfe312576d9f4b53206c8e36e27c4c720e3a6304d6f`.
