# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import pytest

from cuphoton.core.cli import ApplicationContext, get_component, run_component


def _run_cli(argv: list[str]) -> int:
    return run_component("xpois", argv)


def test_component_registry_uses_canonical_namespace() -> None:
    component = get_component("xpois")

    assert component.group == "xpois"
    assert component.import_name == "cuphoton.xpois"


def test_group_program_name_is_used_in_help_output(capsys) -> None:
    rc = _run_cli(["help", "data-inspect"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xpois data-inspect" in captured.out
    assert "--base" in captured.out
    assert "-c FILE, --conf=FILE" not in captured.out


def test_help_for_fit_kernel_command(capsys) -> None:
    rc = _run_cli(["help", "fit-kernel"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xpois fit-kernel" in captured.out
    assert "--reference" in captured.out
    assert "--variance-hdu" in captured.out
    assert "--basis-sigmas" in captured.out
    assert "--backend" in captured.out
    assert "[default: auto]" in captured.out


def test_help_for_benchmark_backends_command_case(capsys) -> None:
    rc = _run_cli(["help", "benchmark-backends"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xpois benchmark-backends" in captured.out
    assert "--backends" in captured.out
    assert "--reference-backend" in captured.out
    assert "--repeats" in captured.out


def test_help_for_evaluate_subtraction_command(capsys) -> None:
    rc = _run_cli(["help", "evaluate-subtraction"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xpois evaluate-subtraction" in captured.out
    assert "--run-dir" in captured.out


def test_help_for_review_bokeh_command(capsys) -> None:
    rc = _run_cli(["help", "review-bokeh"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Usage: cuphoton xpois review-bokeh" in captured.out
    assert "--run-dir" in captured.out


def test_zero_arg_command_rejects_positional_argument(capsys) -> None:
    rc = _run_cli(["data-inspect", "extra"])
    captured = capsys.readouterr()

    assert rc == 1
    assert "invalid number of arguments" in captured.err


def test_top_level_help_exits_zero_to_stdout_case(capsys) -> None:
    rc = _run_cli(["help"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Available subcommands" in captured.out
    assert captured.err == ""


def test_missing_subcommand_help_is_nonzero(capsys) -> None:
    rc = _run_cli(["help", "missing-command"])
    captured = capsys.readouterr()

    assert rc == 1
    assert "Unknown subcommand 'missing-command'" in captured.err


def test_data_inspect_uses_env_shared_hsc_root(
    monkeypatch, tmp_path, capsys
) -> None:
    hsc_root = tmp_path / "shared-hsc"
    hsc_root.mkdir()
    monkeypatch.setenv("CUPHOTON_XPOIS_SHARED_HSC_DIR", str(hsc_root))

    rc = _run_cli(["data-inspect"])
    captured = capsys.readouterr()

    assert rc == 0
    assert str(hsc_root) in captured.out


def test_shared_hsc_discovery_is_domain_owned(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from cuphoton.xpois.data import default_shared_hsc_dir

    hsc_root = tmp_path / "data" / "HSC"
    nested = tmp_path / "work" / "nested"
    hsc_root.mkdir(parents=True)
    nested.mkdir(parents=True)
    monkeypatch.delenv(
        "CUPHOTON_XPOIS_SHARED_HSC_DIR",
        raising=False,
    )
    monkeypatch.chdir(nested)

    assert default_shared_hsc_dir() == hsc_root.resolve()


def test_shared_hsc_fallback_does_not_create_xdg_data_tree(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    from cuphoton.xpois.data import default_shared_hsc_dir

    work_dir = tmp_path / "work"
    work_dir.mkdir()
    data_home = tmp_path / "data-home"
    monkeypatch.delenv(
        "CUPHOTON_XPOIS_SHARED_HSC_DIR",
        raising=False,
    )
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))
    monkeypatch.chdir(work_dir)

    expected = data_home / "cuphoton" / "xpois" / "HSC"
    assert default_shared_hsc_dir() == expected.resolve()
    assert not data_home.exists()


def test_xdg_paths_follow_env_after_import(monkeypatch, tmp_path) -> None:
    state_home = tmp_path / "xdg-state"
    config_home = tmp_path / "xdg-config"
    data_home = tmp_path / "xdg-data"
    monkeypatch.setenv("XDG_STATE_HOME", str(state_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    monkeypatch.setenv("XDG_DATA_HOME", str(data_home))

    context = ApplicationContext.for_component("xpois")
    component_state = (state_home / "cuphoton" / "xpois").resolve()

    assert context.state_dir == component_state
    assert context.workspace_dir == component_state
    assert context.runs_dir == component_state / "runs"
    assert (
        context.config_dir == (config_home / "cuphoton" / "xpois").resolve()
    )
    assert context.data_dir == (data_home / "cuphoton" / "xpois").resolve()
    assert context.log_file == component_state / "logs" / "xpois.log"
    assert not component_state.exists()


def test_cli_smoke_fit_subtract_and_evaluate_case(tmp_path, capsys) -> None:
    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    output_root = tmp_path / "runs"
    arr = [[0.0] * 64 for _ in range(64)]
    for row in range(20, 40):
        for col in range(20, 40):
            arr[row][col] = 1.0
    import numpy as np

    np.save(reference, np.asarray(arr, dtype=np.float64), allow_pickle=False)
    np.save(target, np.asarray(arr, dtype=np.float64), allow_pickle=False)

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--kernel-height",
            "9",
            "--kernel-width",
            "9",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "0",
            "--output-dir",
            str(output_root),
            "--name",
            "fit-run",
            "--backend",
            "cpu",
        ]
    )
    captured = capsys.readouterr()
    fit_summary = json.loads(captured.out)
    assert rc == 0
    assert fit_summary["workflow"] == "fit_kernel"
    assert fit_summary["backend"] == "cpu"
    assert (output_root / "fit-run" / "summary.json").exists()
    assert (output_root / "fit-run" / "artifacts" / "residual.npy").exists()

    rc = _run_cli(
        [
            "subtract",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--kernel-height",
            "9",
            "--kernel-width",
            "9",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "0",
            "--output-dir",
            str(output_root),
            "--name",
            "subtract-run",
        ]
    )
    captured = capsys.readouterr()
    subtract_summary = json.loads(captured.out)
    assert rc == 0
    assert subtract_summary["workflow"] == "subtract"
    assert subtract_summary["requested_backend"] == "auto"
    assert subtract_summary["backend"] == "cpu"
    assert (output_root / "subtract-run" / "summary.json").exists()

    rc = _run_cli(
        [
            "benchmark-backends",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--kernel-height",
            "9",
            "--kernel-width",
            "9",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "0",
            "--output-dir",
            str(output_root),
            "--name",
            "benchmark-run",
            "--backends",
            "cpu",
            "--repeats",
            "2",
            "--warmup",
            "0",
        ]
    )
    captured = capsys.readouterr()
    benchmark_summary = json.loads(captured.out)
    assert rc == 0
    assert benchmark_summary["workflow"] == "benchmark-backends"
    assert benchmark_summary["backends"] == ["cpu"]
    assert benchmark_summary["parity"]["ok"] is True
    assert (output_root / "benchmark-run" / "summary.json").exists()

    rc = _run_cli(
        [
            "evaluate-subtraction",
            "--run-dir",
            str(output_root / "subtract-run"),
        ]
    )
    captured = capsys.readouterr()
    evaluation = json.loads(captured.out)
    assert rc == 0
    assert evaluation["fit_region_pixel_count"] > 0
    assert (output_root / "subtract-run" / "evaluation.json").exists()


def test_cli_benchmark_backends_default_basis_handles_synthetic_cutout(
    tmp_path, capsys
) -> None:
    import numpy as np

    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    variance = tmp_path / "variance.npy"
    output_root = tmp_path / "runs"

    y_coords, x_coords = np.meshgrid(
        np.arange(80, dtype=np.float64),
        np.arange(80, dtype=np.float64),
        indexing="ij",
    )
    image = np.zeros((80, 80), dtype=np.float64)
    for cy, cx, sigma, amp in (
        (22.0, 24.0, 3.0, 90.0),
        (50.0, 44.0, 5.0, 130.0),
        (60.0, 62.0, 2.5, 70.0),
    ):
        image += amp * np.exp(
            -(((x_coords - cx) ** 2) + ((y_coords - cy) ** 2))
            / (2.0 * sigma**2)
        )
    background = 0.05 + 0.01 * x_coords / image.shape[1]
    np.save(reference, image, allow_pickle=False)
    np.save(target, image + background, allow_pickle=False)
    np.save(variance, np.ones_like(image), allow_pickle=False)

    rc = _run_cli(
        [
            "benchmark-backends",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--variance",
            str(variance),
            "--output-dir",
            str(output_root),
            "--name",
            "default-basis-synthetic-cutout",
            "--backends",
            "cpu",
            "--repeats",
            "1",
            "--warmup",
            "0",
        ]
    )

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert rc == 0
    assert summary["workflow"] == "benchmark-backends"
    assert summary["basis"] == [
        {"sigma": 1.5, "degree": 2},
        {"sigma": 3.0, "degree": 1},
        {"sigma": 6.0, "degree": 0},
    ]
    assert summary["parity"]["ok"] is True
    assert summary["fit_pixel_count"]["cpu"] > 0


def test_cli_fit_kernel_supports_auto_stamp_mask(tmp_path, capsys) -> None:
    import numpy as np

    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    variance = tmp_path / "variance.npy"
    y_coords, x_coords = np.meshgrid(
        np.arange(64, dtype=np.float64),
        np.arange(64, dtype=np.float64),
        indexing="ij",
    )
    arr = np.zeros((64, 64), dtype=np.float64)
    for cy, cx, sigma, amp in (
        (18.0, 17.0, 2.0, 120.0),
        (42.0, 39.0, 2.5, 180.0),
        (28.0, 49.0, 1.8, 90.0),
    ):
        arr += amp * np.exp(
            -(((x_coords - cx) ** 2) + ((y_coords - cy) ** 2))
            / (2.0 * sigma**2)
        )
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)
    np.save(variance, np.ones_like(arr, dtype=np.float64), allow_pickle=False)

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--variance",
            str(variance),
            "--kernel-height",
            "9",
            "--kernel-width",
            "9",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "0",
            "--auto-stamp-mask",
            "--auto-stamp-size",
            "15",
            "--auto-stamp-count",
            "2",
            "--auto-peak-percentile",
            "98.0",
            "--output-dir",
            str(tmp_path / "runs"),
            "--name",
            "auto-mask-run",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    summary = json.loads(captured.out)
    assert summary["fit_region"]["kind"] == "auto_stamp_mask"
    assert summary["fit_region"]["selected_count"] == 2
    assert "fit_mask_metadata" in summary["saved"]


def test_cli_fit_kernel_supports_fits_mask_policy_crop_and_auto_stamps(
    tmp_path, capsys
) -> None:
    import numpy as np
    from astropy.io import fits

    reference = tmp_path / "reference.fits"
    target = tmp_path / "target.fits"
    y_coords, x_coords = np.meshgrid(
        np.arange(64, dtype=np.float64),
        np.arange(64, dtype=np.float64),
        indexing="ij",
    )
    arr = np.zeros((64, 64), dtype=np.float64)
    for cy, cx, sigma, amp in (
        (18.0, 17.0, 2.0, 120.0),
        (42.0, 39.0, 2.5, 180.0),
        (28.0, 49.0, 1.8, 90.0),
    ):
        arr += amp * np.exp(
            -(((x_coords - cx) ** 2) + ((y_coords - cy) ** 2))
            / (2.0 * sigma**2)
        )
    variance = np.ones_like(arr, dtype=np.float64)
    mask = np.zeros_like(arr, dtype=np.int64)
    mask[12:16, 12:16] = 1 << 9  # CROSSTALK
    mask[20:24, 20:24] = 1 << 5  # DETECTED only; should survive masklite.

    def write_fits(path):
        image_hdu = fits.ImageHDU(arr, name="IMAGE")
        mask_hdu = fits.ImageHDU(mask, name="MASK")
        for key, bit in (
            ("MP_BAD", 0),
            ("MP_SAT", 1),
            ("MP_INTRP", 2),
            ("MP_CR", 3),
            ("MP_EDGE", 4),
            ("MP_DETECTED", 5),
            ("MP_DETECTED_NEGATIVE", 6),
            ("MP_SUSPECT", 7),
            ("MP_NO_DATA", 8),
            ("MP_CROSSTALK", 9),
            ("MP_NOT_DEBLENDED", 10),
            ("MP_UNMASKEDNAN", 11),
        ):
            mask_hdu.header[key] = bit
        variance_hdu = fits.ImageHDU(variance, name="VARIANCE")
        fits.HDUList(
            [fits.PrimaryHDU(), image_hdu, mask_hdu, variance_hdu]
        ).writeto(path)

    write_fits(reference)
    write_fits(target)

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--reference-hdu",
            "1",
            "--target-hdu",
            "1",
            "--variance",
            str(target),
            "--variance-hdu",
            "3",
            "--mask-policy",
            "hsc-masklite",
            "--auto-stamp-mask",
            "--auto-stamp-size",
            "15",
            "--auto-stamp-count",
            "2",
            "--auto-peak-percentile",
            "98.0",
            "--crop-y0",
            "8",
            "--crop-x0",
            "8",
            "--crop-height",
            "48",
            "--crop-width",
            "48",
            "--kernel-height",
            "9",
            "--kernel-width",
            "9",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "0",
            "--output-dir",
            str(tmp_path / "runs"),
            "--name",
            "fits-auto-mask-run",
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    summary = json.loads(captured.out)
    assert summary["mask_policy"] == "hsc-masklite"
    assert summary["crop"] == {
        "y0": 8,
        "x0": 8,
        "height": 48,
        "width": 48,
    }
    assert summary["fit_region"]["kind"] == "auto_stamp_mask"
    assert summary["fit_region"]["selected_count"] >= 1
    assert "fit_mask_metadata" in summary["saved"]
    assert "input_mask_metadata" in summary["saved"]
    assert summary["input_mask"]["reference_mask_fraction"] > 0.0


def test_cli_review_bokeh_rebuilds_saved_run_case(tmp_path, capsys) -> None:
    pytest.importorskip("bokeh")
    import numpy as np

    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    variance = tmp_path / "variance.npy"
    y_coords, x_coords = np.meshgrid(
        np.arange(64, dtype=np.float64),
        np.arange(64, dtype=np.float64),
        indexing="ij",
    )
    arr = np.zeros((64, 64), dtype=np.float64)
    for cy, cx, sigma, amp in (
        (18.0, 17.0, 2.0, 120.0),
        (42.0, 39.0, 2.5, 180.0),
        (28.0, 49.0, 1.8, 90.0),
    ):
        arr += amp * np.exp(
            -(((x_coords - cx) ** 2) + ((y_coords - cy) ** 2))
            / (2.0 * sigma**2)
        )
    np.save(reference, arr, allow_pickle=False)
    np.save(target, arr, allow_pickle=False)
    np.save(variance, np.ones_like(arr, dtype=np.float64), allow_pickle=False)

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--variance",
            str(variance),
            "--kernel-height",
            "9",
            "--kernel-width",
            "9",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "0",
            "--auto-stamp-mask",
            "--auto-stamp-size",
            "15",
            "--auto-stamp-count",
            "2",
            "--auto-peak-percentile",
            "98.0",
            "--output-dir",
            str(tmp_path / "runs"),
            "--name",
            "review-source-run",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    summary = json.loads(captured.out)
    run_dir = tmp_path / "runs" / "review-source-run"
    review_bokeh_path = run_dir / summary["saved"]["review_bokeh_html"]
    review_bokeh_path.unlink()

    rc = _run_cli(
        [
            "review-bokeh",
            "--run-dir",
            str(run_dir),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    assert "review_bokeh_html" in payload["saved"]
    assert (run_dir / payload["saved"]["review_bokeh_html"]).exists()


def test_prime_preserves_falsey_loaded_options() -> None:
    from cuphoton.xpois import commands as xpois_commands

    command = xpois_commands.FitKernelCommand()
    command.options = {}
    command.load_order = [
        "reference_hdu",
        "target_hdu",
        "background_degree",
        "flux_conserve",
    ]
    command.reference_hdu = 0
    command.target_hdu = 0
    command.background_degree = 0
    command.flux_conserve = False

    primed = command.prime(xpois_commands.SubtractCommand)

    assert primed.reference_hdu == 0
    assert primed.target_hdu == 0
    assert primed.background_degree == 0
    assert primed.flux_conserve is False


def test_fit_kernel_accepts_hdu_zero_and_bad_hdu_is_user_facing(
    tmp_path, capsys
) -> None:
    import numpy as np
    from astropy.io import fits

    reference = tmp_path / "reference.fits"
    target = tmp_path / "target.fits"
    fits.PrimaryHDU(np.ones((32, 32), dtype=np.float64)).writeto(reference)
    fits.PrimaryHDU(np.ones((32, 32), dtype=np.float64)).writeto(target)

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--reference-hdu",
            "0",
            "--target-hdu",
            "0",
            "--kernel-height",
            "15",
            "--kernel-width",
            "15",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "6",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "reference_hdu" not in captured.err.lower()
    assert "target_hdu" not in captured.err.lower()

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--reference-hdu",
            "99",
            "--kernel-height",
            "9",
            "--kernel-width",
            "9",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "0",
            "--output-dir",
            str(tmp_path / "runs-2"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "out of range" in captured.err.lower()


def test_fit_kernel_supports_variance_hdu_selection(tmp_path, capsys) -> None:
    import numpy as np
    from astropy.io import fits

    reference = tmp_path / "reference.fits"
    target = tmp_path / "target.fits"
    variance = tmp_path / "variance.fits"
    arr = np.ones((32, 32), dtype=np.float64)
    fits.PrimaryHDU(arr).writeto(reference)
    fits.PrimaryHDU(arr).writeto(target)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(np.ones((32, 32), dtype=np.float64) * 0.5),
        ]
    ).writeto(variance)

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--variance",
            str(variance),
            "--variance-hdu",
            "1",
            "--kernel-height",
            "15",
            "--kernel-width",
            "15",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "6",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert '"variance_hdu": 1' not in captured.err


def test_fit_kernel_rejects_ambiguous_variance_fits_without_hdu(
    tmp_path, capsys
) -> None:
    import numpy as np
    from astropy.io import fits

    reference = tmp_path / "reference.fits"
    target = tmp_path / "target.fits"
    variance = tmp_path / "variance.fits"
    arr = np.ones((32, 32), dtype=np.float64)
    fits.PrimaryHDU(arr).writeto(reference)
    fits.PrimaryHDU(arr).writeto(target)
    fits.HDUList(
        [
            fits.PrimaryHDU(arr),
            fits.ImageHDU(arr * 0.5),
        ]
    ).writeto(variance)

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--variance",
            str(variance),
            "--kernel-height",
            "9",
            "--kernel-width",
            "9",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "0",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "specify --variance-hdu explicitly" in captured.err


def test_fit_kernel_rejects_variance_hdu_without_variance(
    tmp_path, capsys
) -> None:
    import numpy as np
    from astropy.io import fits

    reference = tmp_path / "reference.fits"
    target = tmp_path / "target.fits"
    arr = np.ones((32, 32), dtype=np.float64)
    fits.PrimaryHDU(arr).writeto(reference)
    fits.PrimaryHDU(arr).writeto(target)

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--variance-hdu",
            "1",
            "--kernel-height",
            "15",
            "--kernel-width",
            "15",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "6",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "variance_hdu requires a variance image path" in captured.err


def test_fit_kernel_rejects_invalid_basis_sigma_tokens(
    tmp_path, capsys
) -> None:
    import numpy as np
    from astropy.io import fits

    reference = tmp_path / "reference.fits"
    target = tmp_path / "target.fits"
    arr = np.ones((32, 32), dtype=np.float64)
    fits.PrimaryHDU(arr).writeto(reference)
    fits.PrimaryHDU(arr).writeto(target)

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--basis-sigmas",
            "1.5,bad",
            "--basis-degrees",
            "0,0",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "--basis-sigmas must contain only numeric values" in captured.err


def test_fit_kernel_rejects_empty_basis_csvs_case(tmp_path, capsys) -> None:
    import numpy as np
    from astropy.io import fits

    reference = tmp_path / "reference.fits"
    target = tmp_path / "target.fits"
    arr = np.ones((32, 32), dtype=np.float64)
    fits.PrimaryHDU(arr).writeto(reference)
    fits.PrimaryHDU(arr).writeto(target)

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--basis-sigmas",
            " , ",
            "--basis-degrees",
            " , ",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "must provide at least one basis term" in captured.err


def test_fit_kernel_rejects_all_non_finite_pixels(tmp_path, capsys) -> None:
    import numpy as np

    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    np.save(reference, np.full((32, 32), np.nan, dtype=np.float64))
    np.save(target, np.full((32, 32), np.nan, dtype=np.float64))

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--kernel-height",
            "9",
            "--kernel-width",
            "9",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "0",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "no finite pixels remain within the fit region" in captured.err


def test_fit_kernel_wraps_filesystem_errors(
    monkeypatch, tmp_path, capsys
) -> None:
    import numpy as np

    from cuphoton.xpois import commands as xpois_commands

    reference = tmp_path / "reference.npy"
    target = tmp_path / "target.npy"
    arr = np.ones((32, 32), dtype=np.float64)
    np.save(reference, arr)
    np.save(target, arr)

    def fail(**kwargs):
        raise PermissionError("permission denied")

    monkeypatch.setattr(xpois_commands, "run_constant_kernel_fit", fail)

    rc = _run_cli(
        [
            "fit-kernel",
            "--reference",
            str(reference),
            "--target",
            str(target),
            "--kernel-height",
            "9",
            "--kernel-width",
            "9",
            "--basis-sigmas",
            "1.5",
            "--basis-degrees",
            "0",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )
    captured = capsys.readouterr()

    assert rc == 1
    assert "permission denied" in captured.err


def test_data_inspect_missing_base_reports_zero_counts(
    tmp_path, capsys
) -> None:
    missing = tmp_path / "missing-hsc"
    rc = _run_cli(["data-inspect", "--base", str(missing)])
    captured = capsys.readouterr()

    assert rc == 0
    assert '"counts": {' in captured.out
    assert '"bundle_files": 0' in captured.out
