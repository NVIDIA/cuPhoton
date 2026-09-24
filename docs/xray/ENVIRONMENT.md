# xRay environment setup

xRay ships in the single `cuphoton` distribution. Run environment and command
operations from the repository root.

## CPU development

```bash
uv sync --locked --extra dev --extra viz
make test-xray
uv run python examples/run_quickstarts.py --component xray --profile cpu
```

## CUDA 13 development

```bash
uv sync --locked --extra dev --extra gpu --extra viz
uv run cuphoton xray doctor
uv run python examples/run_quickstarts.py --component xray --require-gpu
```

The GPU profile targets CUDA 13. `cuphoton xray doctor` reports Python,
CuPy, CUDA visibility, and optional review dependencies; include that output
when reporting environment problems.

## Editable pip installation

```bash
python -m pip install -e '.[dev,viz]'
python -m pip install -e '.[dev,gpu,viz]'
```

Supply your datasets and pass input and output paths explicitly to each command.
