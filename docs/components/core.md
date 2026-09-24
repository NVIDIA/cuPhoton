# Core

`cuphoton.core` owns the command-line, context, path, logging, invariant,
and shared Dragon/MPI execution framework. Component algorithms, datasets,
scientific validation, and output formatting remain in their owning
`cuphoton.*` namespaces.

## What it provides

- the fixed `xdr`, `xfit`, `xpois`, `xscan`, `xrep`, and `xray` component
  registry;
- root, group, and command help through one `cuphoton` entry point;
- class-based command discovery and invariant-backed option validation;
- consistent version, error, and logging behavior; and
- side-effect-free resolution of component XDG config, state, data, run, and
  log paths.

The public surface is pinned by the CLI contract tests. Five groups also
support a component-level `version` command; xDataReader is the exception.

Workflow-specific YAML `--config` options belong to each component.

Shared executor rounds measure `batch_wall_sec` through receipt of all worker
completions. Coordinator artifact audits and scientific output merging follow
that interval; `finalization_sec` measures the component's merge separately.
Inputs retained by an adapter are loaded during worker setup. A round summary
records a completed pass, while the root terminal summary is published after
worker shutdown and the executor's lifecycle checks.

## Public facade

Import shared CLI infrastructure from `cuphoton.core.cli`. Its stable facade
exports:

- `ComponentSpec`, `COMPONENTS`, `COMPONENT_REGISTRY`, and `get_component`;
- `ApplicationContext`;
- `CLI`, `CommandLine`, `Command`, `CommandError`, and
  `InvariantAwareCommand`; and
- scalar, set, path, CSV, sequence, pair, and positional invariant classes.

Repeated values and pairs preserve both declaration order and duplicates.
Variable positionals remain ordered. Component command discovery uses Python
introspection and admits only concrete, public command classes defined
directly in that component's `commands` module. Imported, private, and
abstract classes are excluded; duplicate command names or aliases are errors.

`build_component_cli` and `run_component` also accept an external
`ComponentSpec` directly. This builds and runs the external component while
preserving the public root CLI's fixed registry.

## Common CLI behavior

List groups, then inspect a group or command:

```bash
uv run cuphoton --help
uv run cuphoton xrep --help
uv run cuphoton xrep help reproject-image
```

The equivalent module invocation is `uv run python -m cuphoton`, followed by
the component and command names.

Run-producing commands normally use the component directory beneath
`$XDG_STATE_HOME/cuphoton` when no explicit output root is supplied. Runs and
logs are separated into `runs/` and `logs/`. Pass an explicit output path in
automation to keep artifact locations consistent across environments.

## Extending a component CLI

Define concrete public command classes in the component's `commands` module
and derive their options from Core invariants. Generic parsing, path, logging,
or invariant behavior belongs in Core. Domain-specific options, algorithms,
and scientific validation belong in the component.

A new command should have a unique name and aliases, useful help, tests for
success and invalid input, and a documented artifact contract.

See [Architecture](../architecture.md) and
[Adapting the workflows](../adapting-workflows.md).
