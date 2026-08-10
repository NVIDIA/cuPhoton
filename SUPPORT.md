# Support

cuPhoton is a reference-workflow project with best-effort community support.
It is designed to be inspected and adapted, and does not carry the service or
compatibility guarantees of a supported NVIDIA product.

## Getting help

- Search existing [GitHub issues](https://github.com/NVIDIA/cuPhoton/issues).
- Open a bug report with a minimal reproducer and the output of the relevant
  `<command> --version` and environment checks.
- Open a feature request for a workflow or data-contract proposal.
- Use a pull request for a concrete fix or documented extension.

Include the cuPhoton commit, Python version, operating system, installation
extras, command line, and relevant input shapes. For GPU problems, also include
the GPU model, driver, CUDA runtime, resolved backend, and whether the CPU
profile succeeds. Remove credentials, private paths, proprietary data, and
sensitive metadata from reproductions.

## Scope

Maintainers can help with reproducible defects in this repository and review
well-scoped extensions. They cannot validate the scientific suitability of a
workflow for every instrument, provide access to external datasets, debug
private infrastructure, or commit to response times.

## Security

Do not report vulnerabilities in a public issue. Follow
[`SECURITY.md`](SECURITY.md) for private reporting instructions.
