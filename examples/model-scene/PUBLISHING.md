<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Publishing Model Scene

Read the interactive chapter in the [development guide](https://nvidia.github.io/cuPhoton/main/model-scene.html).
Git contains its output-free source. The versioned publisher renders the saved
GPU results. The following workflow produces a local static chapter for review.

## Build a reviewable artifact

1. Check out the intended committed source revision and keep it unchanged
   through execution and rendering. Install that checkout's cuPhoton with
   xImage and the optional example requirements, as described in README.md.
2. Execute the reference profile in a CUDA 13 environment, retaining the shared
   GPU lock. Use a fresh destination outside the source checkout:

   ```sh
   python examples/model-scene/run.py --profile reference --device cuda \
     --gpu-lock /path/shared-by-all-gpu-users/gpu0.lock \
     --output /path/new-scene-reference
   ```

3. Check `execution.json`, `protocol.json`, `execution.log`, and the study
   diagnostics. Verify source hashes against this checkout, protocol hashes
   against the executed protocol, and artifact hashes against the actual files.
   Inspect the independent-truth endpoints and calibration controls; a successful
   process exit alone does not establish scientific fidelity.
4. In the CPU documentation environment, render the completed notebook with
   `preview.py --executed ... --output ... --source-ref COMMITTED_PUSHED_REF`.
   The command verifies input hashes and invokes the existing postprocessor
   and snapshot checks. Normal CPU guide chapters execute; the GPU notebook
   retains its saved outputs. Build include fragments stay outside the
   published guide resource tree.
5. Test the chapter under `file://` and static HTTP, including a versioned
   `/cuPhoton/<version>/` prefix. Check desktop/mobile, light/dark, keyboard,
   touch, terminology definitions, all selectors, linked pixel values and
   display scales. Block scientific runtime network access and verify that
   there are no WebSocket connections. Audit the HTML and public receipt for
   private paths and restricted material. Confirm GitHub source links resolve.

## Package static files

The complete `_book` directory retains guide navigation and source downloads.
The chapter embeds its plot resources; a local static web server is sufficient.
After verification, package the rendered directory and executed notebook:

```sh
tar -czf /path/model-scene-guide.tar.gz \
  -C /path/new-scene-preview/guide/_book .
cp /path/new-scene-preview/guide/_book/source/model-scene.executed.ipynb \
  /path/model-scene.executed.ipynb
cp /path/new-scene-preview/guide/_book/source/model-scene-execution.json \
  /path/model-scene-execution.json
sha256sum /path/model-scene-guide.tar.gz \
  /path/model-scene.executed.ipynb /path/model-scene-execution.json \
  > /path/model-scene-SHA256SUMS
```

Use the sanitized exported notebook/receipt for public distribution. Retain raw
native workflow records privately when they contain machine-local paths.
Full repeated-trial arrays can be archived separately; the HTML contains selected
experiments and the complete summary diagnostics. Measure the archive and page
sizes before distribution. Preserve numeric precision and avoid embedding
repeated-trial arrays merely to make them accessible from a selector.

To distribute this optional standalone bundle, attach it to a GitHub Release
whose tag contains the exact validated source:

```sh
gh release upload RELEASE_TAG --repo NVIDIA/cuPhoton \
  /path/model-scene-guide.tar.gz /path/model-scene.executed.ipynb \
  /path/model-scene-execution.json /path/model-scene-SHA256SUMS
```

## Update the versioned guide

Use the CUDA reference profile and inspect the local preview as described
above. Follow the [notebook publication procedure](../../docs/README.md#updating-a-notebook-chapter)
with `--name model-scene`, the qualified notebook and its corresponding
execution receipt. Keep private machine paths out of the public inputs and
check the rendered download links for the public notebook and receipt.
