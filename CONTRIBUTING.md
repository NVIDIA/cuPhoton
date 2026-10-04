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
uv sync --locked --extra dev --extra torch --extra viz --extra photometry
make hooks
```

`make hooks` installs Git's pre-commit hook using this clone's development
environment and checks the tracked files. Each contributor runs it in their
own clone; generated files under `.git/hooks` are never committed. For linked
worktrees, install hooks from the primary checkout and keep its `.venv`
available, since Git shares hooks between worktrees. Rerun `make hooks` after
moving the clone or recreating that environment.

For CUDA 13 development:

```bash
uv sync --locked --extra dev --extra gpu --extra viz
```

Use the smallest profile that exercises the change. The experimental `cutile`
extra supports Python 3.12–3.14; use a CUDA 13.2 or newer TileIR compiler
for the supported setup. The `dragon` extra currently supports Python 3.12 and 3.13.

## Checks

Run focused tests for the code you changed, then run the repository checks
before opening a pull request:

```bash
uv lock --check
uv run --locked --extra dev pre-commit run --all-files
make lint
make test-cpu
make build
```

Ruff formats Python and checks imports, common bugs (`B`), and syntax upgrades
for Python 3.12 (`UP`). Mypy checks annotated Python code throughout
`src/cuphoton`; `make typecheck` runs it separately. Scientific and GPU imports
without consistent typing support are skipped, so type checks do not require
CUDA. Unannotated function bodies are not yet checked.

The full pre-commit suite also runs clang-format on C/C++/CUDA sources,
including `.cu` and `.cuh`, and ShellCheck on shell scripts, including
extensionless scripts with a shell shebang. C/C++/CUDA control-flow bodies
require braces. `.clang-format` inserts them automatically, uses an 80-column
limit, and right-aligns macro continuation backslashes at column 80. It keeps
four-space indentation, attached braces, and pointer/reference spacing.
Hook tools are installed automatically by pre-commit; no system clang-format
or ShellCheck is required.
Apply the same brace and macro alignment rules manually to CUDA embedded in
Python strings. Clang-format does not inspect those strings or insert braces
inside preprocessor macro definitions.
Use `make format` for Python fixes and `make ci-lint` for the complete checks.

Validation logs should be clean. If warnings are expected, describe them in the
pull request.

### Python coverage

Run the CPU suite with line and branch coverage from the repository root:

```bash
make test-cpu-coverage
```

This clears previous coverage data and writes `coverage.xml` in Cobertura
format. Coverage includes Python under `src/cuphoton`, including unexecuted
modules, and excludes the generated `_version.py`. Child Python processes
and multiprocessing workers contribute to the report. Compiled C++/CUDA
code, scripts, and examples are outside this Python package measurement.
No minimum percentage is enforced while establishing a baseline.

Regular CI runs without coverage instrumentation. To collect a report for
static analysis, manually run the `coverage` workflow on the desired ref:

```bash
gh workflow run coverage.yml --ref <ref>
```

It runs the CPU suite and synthetic quickstarts with Python 3.12. Maintainers
can add `--field gpu=true` to include the GPU parity tests and GPU quickstarts
on a separate L40G runner. Select a trusted revision before enabling GPU
coverage. The GPU job shares the regular CI GPU queue.

The report job combines the data and uploads `coverage-<commit-SHA>`,
containing `coverage.xml`, the combined `.coverage` database,
`coverage-revision.txt`, and `coverage-profile.txt`. The profile records
whether GPU collection was requested. Artifacts are retained for 14 days.

Download the artifact from the desired CI run before scanning that revision:

```bash
gh run download <run-id> --name coverage-<commit-SHA> --dir .
test "$(git rev-parse HEAD)" = "$(cat coverage-revision.txt)" && \
  echo "Coverage matches this checkout"
```

Run this in a checkout of the recorded revision.
The XML report uses relative source paths so static-analysis tools can
import it from another machine's checkout. Configure the tool's Python
coverage input to use `coverage.xml` before running analysis.

### GPU CI

Pull requests first run CPU and package checks on GitHub-hosted runners.
GPU CI starts when `copy-pr-bot` copies a vetted revision to
`pull-request/<number>` in this repository. Ready PRs from verified NVIDIA
contributors with signed commits sync automatically. Draft PRs need an
explicit maintainer trigger; each new external contribution revision needs
maintainer approval before it can run on a GPU.

After reviewing the current diff, a maintainer can request a bot copy with
`/ok to test <full-head-SHA>`. A maintainer can also push that same reviewed
commit to `pull-request/<number>` directly. Confirm that the copied SHA
matches the current PR head; a passing result for an older revision does
not qualify new changes.

The required `ci-required` check combines CPU, package, and GPU results for
that revision. The initial `ci-pr-checks` result does not satisfy the merge
gate. Pushes to `main` and `0.1.x` also run the GPU checks.

The GPU job uses one L40G with Python 3.12 and the locked CUDA 13 dependencies.
It checks CuPy/PyTorch execution, runs six xFit GPU parity cases, and runs
all five synthetic quickstarts with `--require-gpu`. Missing CUDA support or
skipped parity cases fail the job. JUnit results and quickstart summaries are
uploaded with the tested commit SHA. GPU jobs run one at a time; a new
revision cancels the previous workflow for the same branch.

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
git commit -sS -m "Short imperative summary"
```

- Configure a Git signing key before committing; `-S` adds the cryptographic
  signature and `-s` adds the DCO trailer.
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

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


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
