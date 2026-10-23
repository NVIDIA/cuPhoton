# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest

from cuphoton.core.cli import (
    InvariantError,
    build_component_cli,
    get_component,
    main,
)
from cuphoton.ximage.commands import (
    ArtifactSpecInvariant,
    FiniteFloatInvariant,
    PositiveFloatInvariant,
    UnitIntervalFloatInvariant,
)


class _FakeBundleContext:
    def __init__(self) -> None:
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.closed = True


def _run_cli(argv: list[str], **kwargs) -> int:
    return main(["ximage", *argv], **kwargs)


def test_registered_component_uses_canonical_namespace() -> None:
    assert get_component("ximage").group == "ximage"
    assert get_component("ximage").import_name == "cuphoton.ximage"


def test_cli_uses_shared_component_command_discovery() -> None:
    cli = build_component_cli(get_component("ximage"))

    assert "data-validate" in cli.command_names
    assert "reconstruct-l2" in cli.command_names
    assert "rubin-build-store" in cli.command_names
    assert set(cli._commands_by_token) == set(cli.command_names)


def test_workflow_uses_injected_application_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from cuphoton.ximage import commands

    captured: dict[str, object] = {}

    def fake_run(config_path, **kwargs):
        captured["config_path"] = config_path
        captured.update(kwargs)
        return SimpleNamespace(summary={"status": "ok"})

    monkeypatch.setattr(commands, "run_psf_solve", fake_run)
    output = StringIO()
    error = StringIO()
    state_home = tmp_path / "state"

    assert (
        _run_cli(
            ["psf-solve-adam", "--config", "config.yaml"],
            out=output,
            err=error,
            environ={
                "HOME": str(tmp_path),
                "XDG_STATE_HOME": str(state_home),
            },
        )
        == 0
    )
    assert captured["output_root"] is None
    assert (
        captured["default_output_root"]
        == (state_home / "cuphoton" / "ximage" / "runs").resolve()
    )
    assert json.loads(output.getvalue()) == {"status": "ok"}
    assert error.getvalue() == ""
    assert not state_home.exists()


def test_data_validate_uses_shared_command_error_path(
    monkeypatch,
    capsys,
) -> None:
    from cuphoton.ximage import commands

    bundle = _FakeBundleContext()
    monkeypatch.setattr(
        commands,
        "load_ximage_bundle",
        lambda *args, **kwargs: bundle,
    )
    monkeypatch.setattr(
        commands,
        "validate_bundle",
        lambda bundle: ["images must be rank 3", "psfs are missing"],
    )
    monkeypatch.setattr(
        commands,
        "summarize_bundle",
        lambda bundle: pytest.fail("invalid bundle was summarized"),
    )

    assert _run_cli(["data-validate", "--base", "unused"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "ERROR: xImage bundle validation failed:\n"
        "- images must be rank 3\n"
        "- psfs are missing\n"
    )
    assert bundle.closed is True


def test_data_validate_closes_loaded_bundle_on_success(monkeypatch) -> None:
    from cuphoton.ximage import commands

    bundle = _FakeBundleContext()
    monkeypatch.setattr(
        commands,
        "load_ximage_bundle",
        lambda *args, **kwargs: bundle,
    )
    monkeypatch.setattr(commands, "validate_bundle", lambda value: [])
    monkeypatch.setattr(
        commands,
        "summarize_bundle",
        lambda value: {"images": {"shape": [1, 2, 3, 4]}},
    )
    output = StringIO()
    error = StringIO()

    assert (
        _run_cli(
            ["data-validate", "--base", "converted"],
            out=output,
            err=error,
            environ={},
        )
        == 0
    )
    assert json.loads(output.getvalue()) == {
        "images": {"shape": [1, 2, 3, 4]}
    }
    assert error.getvalue() == ""
    assert bundle.closed is True


def test_data_validate_help_requires_a_complete_bundle() -> None:
    output = StringIO()

    assert _run_cli(["help", "data-validate"], out=output) == 0
    assert "--base" in output.getvalue()
    assert "--artifacts" not in output.getvalue()


def test_data_inspect_closes_loaded_bundle(monkeypatch) -> None:
    from cuphoton.ximage import commands

    bundle = _FakeBundleContext()
    monkeypatch.setattr(
        commands,
        "load_ximage_bundle",
        lambda *args, **kwargs: bundle,
    )
    monkeypatch.setattr(
        commands,
        "summarize_bundle",
        lambda value: {"images": {"shape": [1, 2, 3, 4]}},
    )
    output = StringIO()
    error = StringIO()

    assert (
        _run_cli(
            ["data-inspect", "--store", "converted"],
            out=output,
            err=error,
            environ={},
        )
        == 0
    )
    assert json.loads(output.getvalue()) == {
        "images": {"shape": [1, 2, 3, 4]}
    }
    assert error.getvalue() == ""
    assert bundle.closed is True


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_finite_float_invariant_rejects_nonfinite(value: str) -> None:
    with pytest.raises(InvariantError, match="must be finite"):
        FiniteFloatInvariant.validate(value)


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_positive_float_invariant_rejects_invalid(value: str) -> None:
    with pytest.raises(InvariantError):
        PositiveFloatInvariant.validate(value)


@pytest.mark.parametrize("value", ["0", "-0.1", "1.1", "nan", "inf"])
def test_unit_interval_invariant_rejects_invalid(value: str) -> None:
    with pytest.raises(InvariantError):
        UnitIntervalFloatInvariant.validate(value)


def test_artifact_invariant_rejects_unknown_and_empty_specs() -> None:
    for value in ("bogus", ","):
        with pytest.raises(InvariantError):
            ArtifactSpecInvariant.validate(value)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (
            [
                "data-inspect",
                "--store",
                "data",
                "--artifacts",
                "bogus",
            ],
            "Unknown artifact keys",
        ),
        (
            [
                "rubin-build-reference-catalog",
                "--coadd-fits",
                "coadd.fits",
                "--store",
                "store",
                "--output-csv",
                "catalog.csv",
                "--threshold",
                "nan",
            ],
            "must be finite",
        ),
        (
            [
                "benchmark-devices",
                "--config",
                "config.yaml",
                "--workflow",
                "reconstruct-l2",
                "--catalog-match-radius-px",
                "0",
            ],
            "must be greater than zero",
        ),
        (
            [
                "evaluate-reconstruction",
                "--run-dir",
                "run",
                "--background-min-fraction",
                "1.1",
            ],
            "must be in the interval (0, 1]",
        ),
    ],
)
def test_cli_rejects_invalid_values_before_domain_calls(
    monkeypatch,
    args: list[str],
    message: str,
) -> None:
    from cuphoton.ximage import commands

    def unexpected_call(*args, **kwargs):
        pytest.fail("domain function was called for invalid CLI input")

    monkeypatch.setattr(commands, "load_ximage_bundle", unexpected_call)
    monkeypatch.setattr(
        commands,
        "build_rubin_coadd_reference_catalog",
        unexpected_call,
    )
    monkeypatch.setattr(commands, "benchmark_devices", unexpected_call)
    monkeypatch.setattr(
        commands,
        "evaluate_reconstruction_run",
        unexpected_call,
    )
    output = StringIO()
    error = StringIO()

    assert _run_cli(args, out=output, err=error, environ={}) != 0
    assert output.getvalue() == ""
    assert message in error.getvalue()


def test_help_for_flat_command_uses_docstring_and_invariant_help(
    capsys,
) -> None:
    rc = _run_cli(["help", "psf-solve-adam"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton ximage psf-solve-adam" in captured.out
    assert (
        "Run the packaged h-PSF solve workflow using Adam optimization."
        in captured.out
    )
    assert "--config" in captured.out
    assert "--output-dir" in captured.out
    assert "--log-level" not in captured.out
    assert "-c FILE, --conf=FILE" not in captured.out
    assert captured.err == ""


def test_help_for_benchmark_devices_command(capsys) -> None:
    rc = _run_cli(["help", "benchmark-devices"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton ximage benchmark-devices" in captured.out
    assert "--devices" in captured.out
    assert "--workflow" in captured.out


def test_help_for_evaluate_reconstruction_command(capsys) -> None:
    rc = _run_cli(["help", "evaluate-reconstruction"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton ximage evaluate-reconstruction" in captured.out
    assert "--run-dir" in captured.out
    assert "--coadd-method" in captured.out
    assert "--photometry-threshold" in captured.out
    assert "--catalog-parquet" in captured.out


def test_help_for_rubin_probe_command(capsys) -> None:
    rc = _run_cli(["help", "rubin-probe"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton ximage rubin-probe" in captured.out
    assert "--input" in captured.out
    assert "--read-sample" in captured.out
    assert "--sample-section" in captured.out


def test_help_for_rubin_build_manifest_command(capsys) -> None:
    rc = _run_cli(["help", "rubin-build-manifest"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton ximage rubin-build-manifest" in captured.out
    assert "--input" in captured.out
    assert "--inventory" in captured.out
    assert "--products" in captured.out
    assert "--sample-per-product" in captured.out
    assert "--skip-metadata" in captured.out


def test_help_for_rubin_build_store_command(capsys) -> None:
    rc = _run_cli(["help", "rubin-build-store"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton ximage rubin-build-store" in captured.out
    assert "--manifest" in captured.out
    assert "--input-root" in captured.out
    assert "--align-wcs" in captured.out
    assert "--reference-index" in captured.out
    assert "--mask-policy" in captured.out


def test_help_for_rubin_build_reference_catalog_command(capsys) -> None:
    rc = _run_cli(["help", "rubin-build-reference-catalog"])
    captured = capsys.readouterr()

    assert rc == 0
    assert (
        "Usage: cuphoton ximage rubin-build-reference-catalog" in captured.out
    )
    assert "--coadd-fits" in captured.out
    assert "--store" in captured.out
    assert "--output-csv" in captured.out
    assert "--threshold" in captured.out


def test_help_for_rubin_validate_catalogs_command(capsys) -> None:
    rc = _run_cli(["help", "rubin-validate-catalogs"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton ximage rubin-validate-catalogs" in captured.out
    assert "--input-dir" in captured.out
    assert "--visits-file" in captured.out
    assert "--products" in captured.out


def test_rubin_coadd_semantics_emits_json(
    monkeypatch,
    tmp_path,
    capsys,
) -> None:
    from cuphoton.ximage import commands

    def fake_summary(*args, **kwargs):
        return {
            "coadd_fits_path": str(args[0]),
            "output_path": str(kwargs["output_path"]),
            "rubin_image_format": kwargs["rubin_image_format"],
            "require_coadd_psf": kwargs["require_coadd_psf"],
            "requirements_met": True,
        }

    monkeypatch.setattr(
        commands,
        "summarize_rubin_coadd_semantics",
        fake_summary,
    )

    rc = _run_cli(
        [
            "rubin-coadd-semantics",
            "--coadd-fits",
            str(tmp_path / "coadd.fits"),
            "--rubin-image-format",
            "lsst-images-archive",
            "--output-json",
            str(tmp_path / "semantics.json"),
            "--require-coadd-psf",
            "--require-ok",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    assert '"coadd_fits_path"' in captured.out
    assert '"rubin_image_format": "lsst-images-archive"' in captured.out
    assert '"require_coadd_psf": true' in captured.out
    assert captured.err == ""


def test_rubin_coadd_semantics_require_ok_fails(
    monkeypatch,
    tmp_path,
) -> None:
    from cuphoton.ximage import commands

    monkeypatch.setattr(
        commands,
        "summarize_rubin_coadd_semantics",
        lambda *args, **kwargs: {"requirements_met": False},
    )

    rc = _run_cli(
        [
            "rubin-coadd-semantics",
            "--coadd-fits",
            str(tmp_path / "coadd.fits"),
            "--require-ok",
        ]
    )

    assert rc != 0
