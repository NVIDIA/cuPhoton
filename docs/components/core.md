# Core

`cuphoton.core` owns the command-line, context, path, logging, and invariant
framework shared by every cuPhoton component. It is infrastructure rather than
a science workflow; component algorithms, datasets, validation, and output
formatting remain in their owning `cuphoton.*` namespaces.

## What it provides

- the fixed `xdr`, `xfit`, `xpois`, `xscan`, `xrep`, and `xray` component
  registry;
- root, group, and command help through one `cuphoton` entry point;
- class-based command discovery and invariant-backed option validation;
- consistent version, error, and logging behavior; and
- side-effect-free resolution of component XDG config, state, data, run, and
  log paths.

The fixed public surface has six groups, 89 domain commands, 86 accepted
command aliases, and 754 declared arguments. Five groups also support a
component-level `version` command, for 94 commands when those built-ins are
included; xDataReader is the exception.

Workflow-specific YAML `--config` options remain component concerns. Core does
not define a process-wide INI configuration option.

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
`ComponentSpec` directly. This builds and runs the external component without
registering it in the public root CLI.

## Common CLI behavior

List groups, then inspect a group or command:

```bash
uv run cuphoton --help
uv run cuphoton xrep --help
uv run cuphoton xrep help reproject-image
```

The equivalent module invocation is `uv run python -m cuphoton`. Individual
component packages are not module entry points.

Run-producing commands normally use the component directory beneath
`$XDG_STATE_HOME/cuphoton` when no explicit output root is supplied. Runs and
logs are separated into `runs/` and `logs/`. Pass an explicit output path in
automation when the artifact location must not depend on the environment.

## Extending a component CLI

Define concrete public command classes in the component's `commands` module
and derive their options from Core invariants. Generic parsing, path, logging,
or invariant behavior belongs in Core. Domain-specific options, algorithms,
and scientific validation belong in the component.

A new command should have a unique name and aliases, useful help, tests for
success and invalid input, and a documented artifact contract.

See [Architecture](../architecture.md) and
[Adapting the workflows](../adapting-workflows.md).
