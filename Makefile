# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

.PHONY: sync sync-gpu sync-cutile lock lock-check lint format test test-cpu test-core test-xdr test-xfit test-xfit-real test-xpois test-xscan test-xrep test-xray test-gpu clean-dist build package-check release-check ci-lint ci-test-cpu hooks

CPU_EXTRAS = --extra dev --extra torch --extra viz
GPU_EXTRAS = --extra dev --extra gpu --extra viz
CORE_EXTRAS = --extra dev
VIZ_EXTRAS = --extra dev --extra viz
UV_RUN = uv run --locked

sync:
	uv sync --locked $(CPU_EXTRAS)

sync-gpu:
	uv sync --locked $(GPU_EXTRAS)

sync-cutile:
	uv sync --locked --python 3.12 $(GPU_EXTRAS) --extra cutile

lock:
	uv lock

lock-check:
	uv lock --check

lint:
	$(UV_RUN) --extra dev ruff check .
	$(UV_RUN) --extra dev ruff format --check .

format:
	$(UV_RUN) --extra dev ruff check --fix .
	$(UV_RUN) --extra dev ruff format .

test:
	$(UV_RUN) $(CPU_EXTRAS) pytest

test-cpu:
	CUDA_VISIBLE_DEVICES= CUPHOTON_XREP_TORCH_DEVICE=cpu $(UV_RUN) $(CPU_EXTRAS) pytest

test-core:
	$(UV_RUN) $(CORE_EXTRAS) pytest tests/core

test-xdr:
	$(UV_RUN) $(CORE_EXTRAS) pytest tests/xdr

test-xfit:
	$(UV_RUN) $(CORE_EXTRAS) pytest tests/xfit

test-xfit-real:
	$(UV_RUN) $(CORE_EXTRAS) pytest -rs -m real_data tests/xfit/test_real_data.py

test-xpois:
	$(UV_RUN) $(VIZ_EXTRAS) pytest tests/xpois

test-xscan:
	$(UV_RUN) $(CPU_EXTRAS) pytest tests/xscan

test-xrep:
	$(UV_RUN) $(CPU_EXTRAS) pytest tests/xrep

test-xray:
	$(UV_RUN) $(VIZ_EXTRAS) pytest tests/xray

test-gpu:
	$(UV_RUN) $(GPU_EXTRAS) pytest

clean-dist:
	rm -rf dist

build: clean-dist
	uv build

package-check: build
	uvx --isolated --from twine==6.2.0 twine check dist/*

release-check:
	$(MAKE) lock-check
	$(MAKE) ci-lint
	$(MAKE) ci-test-cpu
	$(MAKE) package-check

ci-lint:
	$(MAKE) lint
	$(UV_RUN) --extra dev pre-commit run --all-files --show-diff-on-failure

ci-test-cpu: test-cpu

hooks:
	$(UV_RUN) --extra dev pre-commit install
	$(UV_RUN) --extra dev pre-commit run --all-files --show-diff-on-failure
