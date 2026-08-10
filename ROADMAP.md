# Roadmap

cuPhoton v0.1.0 establishes runnable, inspectable reference workflows rather
than a compatibility-stable library. Planned work after v0.1.0 is organized
around reproducibility and easier institutional adaptation.

## Near term

- Expand CUDA 13 continuous integration beyond smoke coverage.
- Add representative performance baselines for supported GPU architectures.
- Exercise wheel and source-distribution installs across all documented
  profiles.
- Improve synthetic fixtures so more workflow branches can be validated
  without external datasets.
- Add focused examples for custom masks, WCS grids, detector layouts, and
  metadata adapters.

## Packaging

- Publish pip packages after release automation and provenance checks are
  established.
- Add conda packaging with the same dependency-profile boundaries.
- Keep experimental toolchain integrations, such as cuTile, isolated from the
  core install.

## Interfaces

- Stabilize a small set of Python data and workflow contracts based on use by
  external research teams.
- Continue to treat component CLIs and structured artifacts as the primary
  reproducible interfaces.
- Add migrations when a widely used artifact schema changes.

Roadmap items are intentions, not commitments. Priorities are set through
maintainer review and public issue discussion.
