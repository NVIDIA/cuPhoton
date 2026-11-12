# Developer-tool instructions

Scope: the independent `dev/` project and its documentation publishing workflows.

- Read [the runbook](README.md) before changing publication behavior. Install
  this project explicitly with `uv sync --project dev --locked`; root installs
  and the product's existing `dev` extra must remain independent.
- Reuse `cuphoton.core` command, invariant and context machinery. Keep the
  product component registry fixed and the optional route lazy. Product wheels,
  sdists, Conda packages and Git source archives must exclude this project.
- Keep orchestration and postprocessing on main. Render exact target source
  with its lockfile in a separate environment; never replace historical
  chapters or edit an existing tag. Authored guide URLs continue to name main.
- Preserve the `docs.yml` workflow name and string `tag` dispatch input. Release
  branches inherit only the forwarding hook; verify it is present before each
  future release cut. Publishing packages is not the GitHub Release signal.
- Every publication must validate formal-release eligibility, retained hashes,
  tag identity and the complete catalog. Hold the shared publication lock from
  reading the archive through native Pages deployment. Keep PR jobs read-only.
- Keep transformation fixtures synchronized with guide markup, install examples,
  link structure and search output. Unexpected version-sensitive markup should
  fail validation rather than silently publish stale main instructions.
- Update `scripts/ci/dev_changes.py` when adding a dependency or transformation
  input. Update and check `dev/uv.lock` after changing root dependency metadata;
  do not add this project to the root lock, groups, extras or workspace.
- Run developer tests and type checks separately from product tests and coverage.
  Exercise real main and release renders, navigation, search, themes and bookmark
  redirects when changing transformations or publication behavior. Inspect
  deployed URLs and hashes before reporting a rollout complete.
- Follow the repository's Apache-2.0 SPDX/copyright, signed-commit and DCO rules
  for source changes. Generated `gh-pages` output uses GitHub Actions bot
  commits and is exempt from source identity, signature and DCO requirements.
  Keep local environments, output trees, review notes and credentials untracked.
