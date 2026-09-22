## Summary

Describe the change and the affected `cuphoton.*` component or workflow.

## Validation

List the commands run and their results.

```bash
uv lock --check
uv run --locked --extra dev pre-commit run --all-files
make lint
make test-cpu
```

## Repository hygiene

- [ ] No credentials, private paths, local datasets, notebook outputs, or generated run artifacts were added.
- [ ] Large binary artifacts are excluded or intentionally tracked through an approved storage plan.
- [ ] Security-sensitive information is not included.
- [ ] User-visible behavior changes are documented.
- [ ] New dependencies and their licenses are documented.
