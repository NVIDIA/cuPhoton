<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Publishing the Model Sky HTML

The repository contains the output-free notebook and its helpers. The rendered
version includes cell outputs, six interactive figures, terminology hovers,
expandable explanations, equations and the guide's light/dark themes.

Read the chapter in the [development guide](https://nvidia.github.io/cuPhoton/main/model-sky.html).
The publisher renders it from a reviewed GPU execution. To reproduce the study
or qualify a new execution locally, follow the steps below.

## Build and review an export

1. Check out the committed revision you intend to publish. For a release,
   use a tag that contains `examples/model-sky/`. Keep the checkout unchanged
   through execution and rendering.
2. Prepare a CUDA 13 environment with this checkout's cuPhoton and the
   additional dependencies in `requirements.txt`. Use a separate CPU docs
   environment with Quarto, following [the docs instructions](../../docs/README.md).
3. Execute the notebook, including both optional investigations, then render
   it inside the guide. Replace the environment and output paths below:

   ```sh
   export MODEL_SKY_EXPORT=/absolute/path/outside/checkout/model-sky-export
   /path/to/gpu-env/bin/python examples/model-sky/run.py \
     --output "$MODEL_SKY_EXPORT/reference"
   /path/to/docs-env/bin/python examples/model-sky/preview.py \
     --executed "$MODEL_SKY_EXPORT/reference/model-sky.executed.ipynb" \
     --output "$MODEL_SKY_EXPORT/render" \
     --source-ref "$(git rev-parse HEAD)"
   python -m http.server 8796 --bind 0.0.0.0 \
     --directory "$MODEL_SKY_EXPORT/render/guide/_book"
   ```

   The two output destinations must be new or empty. The renderer verifies the
   successful execution receipt and source hashes before reusing outputs.
   `--source-ref` controls the visible GitHub notebook link; that revision must
   have been pushed for the link to work.
4. Open `http://localhost:8796/model-sky.html`. Check all six plots, glossary
   hovers, expanded explanations, themes, mobile layout and source links.
   Review the measurements and `reference/execution.json` as well as the page.

## Share a downloadable HTML bundle

Package the complete rendered directory so guide navigation and source
downloads work. Although the Model Sky page embeds its plot resources, its
navigation and downloads refer to other files in this directory.

```sh
tar -czf "$MODEL_SKY_EXPORT/model-sky-guide.tar.gz" \
  -C "$MODEL_SKY_EXPORT/render/guide/_book" .
cp "$MODEL_SKY_EXPORT/reference/model-sky.executed.ipynb" \
  "$MODEL_SKY_EXPORT/model-sky.executed.ipynb"
cp "$MODEL_SKY_EXPORT/reference/execution.json" \
  "$MODEL_SKY_EXPORT/model-sky-execution.json"
(
  cd "$MODEL_SKY_EXPORT"
  sha256sum model-sky-guide.tar.gz model-sky.executed.ipynb \
    model-sky-execution.json > model-sky-SHA256SUMS
)
```

After reviewing the bundle, upload these four files as assets of the intended
GitHub Release. Use a release whose tag matches the source used above:

```sh
gh release upload RELEASE_TAG --repo NVIDIA/cuPhoton \
  "$MODEL_SKY_EXPORT/model-sky-guide.tar.gz" \
  "$MODEL_SKY_EXPORT/model-sky.executed.ipynb" \
  "$MODEL_SKY_EXPORT/model-sky-execution.json" \
  "$MODEL_SKY_EXPORT/model-sky-SHA256SUMS"
```

Replace `RELEASE_TAG` with that existing release's tag. The command uploads
assets; it does not create or publish a software release. Keep existing assets
intact rather than replacing them. The generated FITS inputs are reproducible
from the example and are not required to view the HTML. Recipients can extract
the bundle and serve it with `python -m http.server`.

## Update the versioned guide

Use the full reference execution and matching receipt from the review above.
The [notebook publication procedure](../../docs/README.md#updating-a-notebook-chapter)
records those inputs under `model-sky` and pins their digests in the guide
manifest. Build the guide for that commit and check all six interactive plots
and the source downloads. The versioned publisher renders the saved outputs
for each revision that declares the chapter.
