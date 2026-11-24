<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Publishing Model Alerts

Read the chapter in the [development guide](https://nvidia.github.io/cuPhoton/main/model-alerts.html).
Its interactive review uses saved results in the browser. Follow the local
export workflow to inspect a full reference execution before recording it.

## Execute, verify and render

1. Select a complete source revision and preserve it throughout execution.
   Provision a CUDA 13 environment as described in [README.md](README.md).
   Run `run.py --profile full-reference` into a new directory outside Git,
   using the shared GPU lock. Keep earlier failed/invalidated runs intact.
2. Inspect `execution.json`, `study/freeze.json`, `study/report.json` and
   scientific control outputs. Verify every `source_sha256` against the
   matching example source, and every `output_sha256` against the named
   artifact using SHA-256. The executed notebook must contain every cell,
   warning and figure with no execution errors. A quick-smoke receipt is API
   evidence only, even when execution succeeds.
3. Use a separate CPU documentation environment with current Quarto, following
   [the guide build instructions](../../docs/README.md). Install this example's
   optional requirements and the repository's optional developer tools there.
   Export into another new directory:

   ```sh
   /path/to/docs-env/bin/python examples/model-alerts/preview.py \
     --executed /absolute/path/full-reference/model-alerts.executed.ipynb \
     --output /absolute/path/new-model-alerts-render \
     --source-ref PUSHED_SOURCE_COMMIT
   ```

   Replace `PUSHED_SOURCE_COMMIT` with the exact reviewed source SHA. The
   renderer verifies executed/source hashes before reusing GPU cell outputs,
   executes the ordinary CPU guide chapters, applies the current developer
   version postprocessor and runs `docs-check`. Includes remain build inputs
   outside published resource trees.
   `--allow-smoke` enables an explicitly labeled engineering preview only.
4. Serve only the rendered book, never its parent containing data/weights:

   ```sh
   python -m http.server 8797 --bind 0.0.0.0 \
     --directory /absolute/path/new-model-alerts-render/guide/_book
   ```

   Open `http://localhost:8797/model-alerts.html`. Verify every control, term
   hover/focus/touch/Escape behavior, equations and light/dark theme. Check
   exact pixel values and unchanged numerical data under display changes.
   Repeat with scientific runtime network access blocked, zero WebSockets,
   `file://` use, mobile/desktop and DPR 1/2, and a nested
   `/cuPhoton/main/model-alerts.html` route. Verify visible GitHub source links.
5. Audit final HTML and public receipts for private paths, account/machine
   identifiers, unintended remote dependencies and restricted assets. Measure
   source/artifact bytes and inspect arrays for avoidable duplication before
   considering any precision change. Do not ship training data or weights
   inside HTML.

## Package reviewed static assets

After qualification, package the entire book so navigation and downloads work:

```sh
export CUPHOTON_MODEL_ALERTS_EXPORT=/absolute/path/reviewed-export
mkdir "$CUPHOTON_MODEL_ALERTS_EXPORT"
tar -czf "$CUPHOTON_MODEL_ALERTS_EXPORT/model-alerts-guide.tar.gz" \
  -C /absolute/path/model-alerts-render/guide/_book .
cp /absolute/path/full-reference/model-alerts.executed.ipynb \
  "$CUPHOTON_MODEL_ALERTS_EXPORT/"
cp /absolute/path/full-reference/execution.json \
  "$CUPHOTON_MODEL_ALERTS_EXPORT/model-alerts-execution.json"
cp /absolute/path/full-reference/model-alerts-source.zip \
  "$CUPHOTON_MODEL_ALERTS_EXPORT/"
(
  cd "$CUPHOTON_MODEL_ALERTS_EXPORT"
  sha256sum model-alerts-guide.tar.gz model-alerts.executed.ipynb \
    model-alerts-execution.json model-alerts-source.zip > model-alerts-SHA256SUMS
)
```

Verify checksums after copying. An archive extracted locally must still open
without an application server. Python's static HTTP server is optional.

To distribute this optional standalone bundle, select an existing GitHub
Release whose tag matches the source used for execution:

```sh
gh release upload RELEASE_TAG --repo NVIDIA/cuPhoton \
  "$CUPHOTON_MODEL_ALERTS_EXPORT/model-alerts-guide.tar.gz" \
  "$CUPHOTON_MODEL_ALERTS_EXPORT/model-alerts.executed.ipynb" \
  "$CUPHOTON_MODEL_ALERTS_EXPORT/model-alerts-execution.json" \
  "$CUPHOTON_MODEL_ALERTS_EXPORT/model-alerts-source.zip" \
  "$CUPHOTON_MODEL_ALERTS_EXPORT/model-alerts-SHA256SUMS"
```

## Update the versioned guide

Use the reviewed full-reference output and its matching execution receipt.
Follow the [notebook publication procedure](../../docs/README.md#updating-a-notebook-chapter)
with `--name model-alerts`. Validate the gallery, classification controls and source
downloads in the guide built from the updated manifest.
