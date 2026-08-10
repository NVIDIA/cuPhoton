# Contributing to cuPhoton

Thank you for contributing to cuPhoton. This project publishes reference
workflows: changes should remain readable, reproducible, and practical for a
research team to adapt rather than assume one institution's environment.

## Issue Tracking

Open an issue before starting a substantial feature, dependency change, or
artifact-schema change. Small bug fixes and documentation corrections may go
directly to a pull request.

For non-security bug reports, feature requests, and development tasks,
include:

- the affected `cuphoton.*` component;
- the Python, CUDA, driver, and OS versions;
- the command or workflow that failed;
- the expected behavior and observed behavior;
- a minimal reproducer when possible.

Do not report security vulnerabilities through GitHub. Follow `SECURITY.md`.

## Coding Guidelines

All Python code ships in one distribution and must live under `cuphoton.*`.
Shared CLI, application-context, logging, and invariant framework code belongs
in `cuphoton.core`. Workflow-specific configuration remains in the component
that owns the workflow. Do not add top-level Python packages.

Keep pull requests concise:

- address one concern per pull request;
- avoid committing commented-out code;
- avoid unrelated refactors;
- add or update tests when behavior changes;
- update README or usage documentation for user-visible changes;
- include documentation and tests for any new component, command, or workflow;
- keep generated outputs, credentials, local paths, datasets, and notebook
  checkpoints out of the repository unless maintainers explicitly approve
  the content and storage plan.

All contributed source, configuration, and script files must carry the
repository's Apache-2.0 SPDX license identifier and NVIDIA copyright header
unless the maintainers approve a file-type-specific exception.

## Development Environment

Use uv for development environments and dependency locking. The supported GPU
profile is CUDA 13.

```bash
uv sync --locked --extra dev --extra torch --extra viz
uv run --locked --extra dev pre-commit install
```

For CUDA 13 development:

```bash
uv sync --locked --extra dev --extra gpu --extra viz
```

Use the smallest profile that exercises the change. The `cutile` extra is
experimental and supports Python 3.12 and 3.13 only.

## Checks

Run focused tests for the code you changed, then run the repository checks
before opening a pull request:

```bash
uv lock --check
uv run --locked --extra dev pre-commit run --all-files
make lint
make test-cpu
uv build
```

Validation logs should be clean. If warnings are expected, describe them in the
pull request.

## Pull Requests

Developer workflow for code contributions:

1. Fork the repository or create a topic branch.
2. Commit focused changes to the topic branch.
3. Run the checks listed above.
4. Open a pull request targeting `main`.
5. Include the issue number, summary, validation results, and residual risks in
   the pull request description.
6. Mark incomplete pull requests as draft or prefix the title with `[WIP]`.

Maintainers may request dependency, licensing, security, scientific, or
performance review before accepting a change.

## Commit Requirements

- Make small, topical commits.
- Sign commits with a GitHub-verifiable signature.
- Include a Developer Certificate of Origin sign-off on commits:

```bash
git commit -s -m "Short imperative summary"
```

- Write commit titles in imperative mood.
- Target the `main` branch.

## Developer Certificate of Origin

By making a contribution to this project, you certify the Developer Certificate
of Origin. The canonical DCO text is published at
https://developercertificate.org/ and included below:

```text
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this license
document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

## Review Reproducibility

When adding a workflow, persist enough metadata for another maintainer to
reproduce the run visually and numerically. Include input identities and
shapes, command lines, effective configuration, random seeds, package
revisions, selected backend/device, and hardware details where they affect the
result. Do not persist credentials or machine-specific private paths.

Document the input and output contract, provide a synthetic or redistributable
smoke path when feasible, and state the numerical tolerance or scientific
criterion used for validation. Performance claims should report warmup,
repetition, synchronization, and hardware details.
