# Fresh-launch benchmark campaigns

`run_campaign.py` runs existing local benchmark adapters in fresh processes.
The caller prepares inputs, dependencies, GPU reservations and resource limits.
The runner does not allocate resources or change scientific defaults.

```sh
python examples/benchmarks/run_campaign.py --manifest campaign.json --output-dir new-results --dry-run
python examples/benchmarks/run_campaign.py --manifest campaign.json --output-dir new-results
```

Dry-run validates declarations and prints the resolved schedule, original
manifest SHA256 and resolved plan SHA256 without running commands or creating
output. Execution saves that plan and refuses an existing output directory.
Each block rotates treatment order; successive complete cycles reverse it.
Every treatment occupies each position once per complete cycle. A partial
cycle is recorded as such in the schedule; it is not completely balanced.
Six fresh blocks are the default and minimum for explicit `headline` runs.
An omitted `experiment_kind` defaults to `pilot`; pilots may use fewer blocks
and are never headline-eligible.

A manifest contains:

```json
{
  "schema_version": 1,
  "experiment_kind": "headline",
  "blocks": 6,
  "timeout_seconds": 600,
  "comparison": {"kind": "matched", "feature": "implementation"},
  "workload": {"name": "declared workload"},
  "output_contract": {"endpoint": "validated saved arrays"},
  "cache": {"policy": "declared by adapter"},
  "disclosure": {"classification": "public"},
  "environment": {"dependency_manifest_sha256": "..."},
  "input_hashes": {"dataset": "..."},
  "config": {"iterations": 20},
  "treatments": [{
    "name": "reference",
    "cwd": "/absolute/source/worktree",
    "argv": ["/absolute/venv/bin/python", "adapter.py", "run", "{launch_dir}"],
    "validation_argv": ["/absolute/venv/bin/python", "adapter.py", "validate", "{launch_dir}"],
    "source_sha": "0123456789abcdef0123456789abcdef01234567",
    "backend": "numpy",
    "expected_completed_work": 10,
    "env": {"OMP_NUM_THREADS": "1"}
  }]
}
```

`environment`, `input_hashes`, `config` and treatment `env` are optional.
Relative `cwd` resolves from the manifest directory. Commands are argv lists,
never shell expressions; only literal `{launch_dir}` substitution occurs.
The current environment is inherited, and only declared overrides are saved.
Do not put secrets in manifests. Source, dependency and input hashes are
declarations: adapters must verify actual loaded source and bytes, and record
relevant effective runtime settings without dumping the whole environment.

The workload command writes `result.json` in its unique launch directory:

```json
{
  "status": "ok",
  "source_sha": "0123456789abcdef0123456789abcdef01234567",
  "actual_backend": "numpy",
  "completed_work": 10,
  "quality_checks": [{"name": "completed outputs", "passed": true}],
  "input_identities": {"dataset_sha256": "..."},
  "output_identities": {"array_sha256": "..."},
  "warm_rounds": [{"elapsed_seconds": 1.25}, {"elapsed_seconds": 1.19}]
}
```

The workload must leave `validation.json` absent. The separate validation
command independently checks saved outputs and creates
`validation.json` with `status: "ok"` and a nonempty `quality_checks` list in
the same format. Optional validator `source_sha` and `completed_work` must
match the treatment. The runner hashes `result.json` after the workload and
requires the validator to leave those bytes unchanged; both receipt hashes
are saved with the launch. Include measurements, tolerances and identities
as additional fields. This must perform scientific validation, not merely
repeat the workload's success flag. Both receipts are retained in full.
Missing/failed checks, mismatched source/backend/work count, invalid counts
and nonfinite, zero or negative measured warm times reject the launch.
Input/output identity objects are required. An adapter decides its
completed-work unit explicitly in `output_contract`; failed work must remain
in its denominator.

Adapters own warmup and persistent measured rounds; existing cuPhoton
`BenchmarkOptions` can provide those rounds. `warm_rounds` contains measured
rounds only (optional `phase` must equal `measure`). It may be empty for a
single-shot application. If any round supplies `completed_work`, every round
must supply a positive integer and their sum must match the launch total.
Optional `workload.measured_rounds` must match the number of reported rounds.
The runner reports each launch's warm median and round count, marks differing
round counts in the treatment summary, and keeps rounds nested; they are not
independent launch observations.

Headline wall time begins immediately before workload startup and ends after
the validation process and receipt checks. It includes imports, compilation,
input preparation performed by the adapter, computation and declared output
publication. Stage logs and elapsed seconds remain separate. A shared launch
timeout covers both commands; timeout sends TERM then KILL to the owned
process group, allowing up to two seconds after each signal while polling
and reaping the owned child.
The runner also checks the group after its parent exits normally, allowing
up to 250 milliseconds within the launch deadline for helpers to finish.
This wait is included in stage timing and recorded as `group_linger_seconds`.
Surviving children are terminated and reject the launch. If the group still
exists after cleanup, including unreaped zombies, cleanup is conservatively marked
incomplete and subsequent launches are skipped. SIGINT, SIGTERM or SIGHUP
to the CLI also terminates that group, saves the interrupted launch receipt
and stops without starting further launches. Repeated stop signals do not interrupt
cleanup or receipt writes; the first signal determines the exit status.
Adapters must keep children in that group and close their output;
they must not daemonize, resume old results or skip measured work.

`launches.json` is updated after every attempt; `summary.json` retains planned,
attempted, failed and skipped counts, realized treatment-position counts, and
per-launch identities. A treatment with any failed or skipped launch gets no
accepted wall median. Successful pilots remain pilots.
Reported medians are descriptive; no confidence interval or significance
estimate is inferred from a small cohort. Complete numerical and disclosure
qualification remains the campaign owner's responsibility.
