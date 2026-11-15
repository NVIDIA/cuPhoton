<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Publishing the Model Sky HTML

The repository contains the output-free notebook and its helpers. The rendered
version includes cell outputs, six interactive figures, terminology hovers,
expandable explanations, equations and the guide's light/dark themes.

This example currently provides a local export workflow. Adding it to the
automatically published, versioned guide requires the integration described
below. Merging the example alone does not add Model Sky to GitHub Pages.

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

## Publish a versioned GitHub Pages chapter

Complete these changes in the developer tooling before enabling automatic
publication:

1. Stage Model Sky from the selected source revision in `cuphoton dev
   docs-build`. Accept an executed notebook with matching example hashes,
   cuPhoton source revision and execution dependencies. Reuse its saved outputs
   while executing ordinary CPU guide chapters; the builder's current global
   `--execute` would try to rerun the GPU notebook in its CPU environment.
2. Pass the rendered guide through the existing version postprocessor and
   `docs-check`. Retain the chapter's HTML includes as build inputs only.
   Existing source-link conversion should map `main` links to release tags.
3. Add durable storage for the large rendered page and execution artifacts.
   Keep a manifest of exact asset locations and hashes in `gh-pages`; restore
   declared assets into the temporary deployment directory for every retained
   version, then validate the complete site. The current publisher commits
   every rendered file, so uploading Release assets alone does not avoid Git
   growth. Main needs durable assets keyed to its source revision as well;
   expiring PR artifacts and a mutable latest-release fallback are insufficient.
4. Validate a main build, a release containing the example, and an older
   release without it. Check source links, navigation, search, version
   switching, missing/corrupted assets and preservation of earlier snapshots.
   Keep rendering, archive writing and Pages deployment in their existing
   separate jobs.
5. Once that integration is merged and validated, use the normal main-push
   publication workflow. For an eligible published release, use the existing
   `cuphoton dev docs-publish --ref RELEASE_TAG` command when a manual dispatch
   is needed. Verify the served chapter and source link after deployment.

Expected routes are `main/model-sky.html` and, for releases that contain this
example, `vX.Y.Z/model-sky.html` beneath the cuPhoton Pages site. Keep generated
notebooks, data and HTML out of source commits throughout this process.
