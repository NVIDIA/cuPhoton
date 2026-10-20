# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import copy
from pathlib import Path

import pytest
import yaml
from transient_chain import config

HERE = Path(__file__).resolve().parent.parent


def test_shipped_config_loads_and_fills_defaults():
    cfg = config.load_config(HERE / "config.yaml")
    assert set(cfg) == set(config.SCHEMA)
    assert cfg["io"]["reader"] == "auto" and cfg["survey"]["name"] == "rubin"
    assert cfg["xfit"]["solver"]["initial_damping"] == 1.0e-3
    assert cfg["align"]["star_box_px"] == 15


def test_unknown_block_and_key_are_rejected(tmp_path):
    raw = yaml.safe_load((HERE / "config.yaml").read_text())
    bad = copy.deepcopy(raw)
    bad["xfit"]["model"] = "gaussian"
    with pytest.raises(ValueError, match="unknown keys \\['model'\\]"):
        config.validate_config(bad)
    bad = copy.deepcopy(raw)
    bad["extra"] = {}
    with pytest.raises(ValueError, match="unknown blocks"):
        config.validate_config(bad)


def test_required_keys_and_value_checks():
    raw = yaml.safe_load((HERE / "config.yaml").read_text())
    bad = copy.deepcopy(raw)
    del bad["detect"]["threshold_sigma"]
    with pytest.raises(
        ValueError, match="detect.threshold_sigma' is required"
    ):
        config.validate_config(bad)
    bad = copy.deepcopy(raw)
    bad["stamps"]["stamp_size"] = 50
    with pytest.raises(ValueError, match="odd"):
        config.validate_config(bad)
    bad = copy.deepcopy(raw)
    bad["io"]["reader"] = "cfitsio"
    with pytest.raises(ValueError, match="io.reader must be one of"):
        config.validate_config(bad)
    bad = copy.deepcopy(raw)
    bad["classify"]["inference_policy"] = "Strict"
    with pytest.raises(ValueError, match="inference_policy"):
        config.validate_config(bad)
    bad = copy.deepcopy(raw)
    bad["classify"]["noise_reference_sigma"] = None
    with pytest.raises(ValueError, match="noise_reference_sigma is required"):
        config.validate_config(bad)
    bad = copy.deepcopy(raw)
    bad["xfit"]["solver"] = {"max_evaluations": 0}
    with pytest.raises(ValueError, match="max_evaluations"):
        config.validate_config(bad)


def test_overrides_parse_yaml_values_and_nest():
    cfg = config.load_config(
        HERE / "config.yaml",
        [
            "io.reader=astropy",
            "align.star_finder=daofind",
            "xfit.solver.max_evaluations=400",
            "classify.apply_noise_match=false",
        ],
    )
    assert (
        cfg["io"]["reader"] == "astropy"
        and cfg["align"]["star_finder"] == "daofind"
    )
    assert (
        cfg["xfit"]["solver"]["max_evaluations"] == 400
        and cfg["classify"]["apply_noise_match"] is False
    )
    with pytest.raises(ValueError, match="block.key=value"):
        config.load_config(HERE / "config.yaml", ["reader"])


def test_config_digest_ignores_checkpoint_path():
    cfg = config.load_config(HERE / "config.yaml")
    a = config.config_sha256(cfg)
    cfg["classify"]["checkpoint_dir"] = "/some/host/path"
    assert config.config_sha256(cfg) == a
    cfg["detect"]["threshold_sigma"] = 6.0
    assert config.config_sha256(cfg) != a


def test_config_revalidates_after_checkpoint_path_injection(tmp_path):
    cfg = config.load_config(HERE / "config.yaml")
    cfg["classify"]["checkpoint_dir"] = str(
        tmp_path
    )  # what options_payload injects before the worker sees it
    again = config.validate_config(cfg)
    assert again["classify"]["checkpoint_dir"] == str(tmp_path)
    cfg["classify"]["checkpoint_dir"] = 5
    with pytest.raises(ValueError, match="checkpoint_dir must be str"):
        config.validate_config(cfg)


def test_fit_mode_is_difference_only():
    with pytest.raises(ValueError, match="mode"):
        config.load_config(HERE / "config.yaml", ["xfit.mode=search"])


@pytest.mark.parametrize("value", ["[1.5]", "[-1]", "[true]", "[]"])
def test_basis_degrees_reject_invalid_values(value):
    with pytest.raises(ValueError, match="basis_degrees"):
        config.load_config(
            HERE / "config.yaml", [f"subtract.basis_degrees={value}"]
        )


def test_basis_degrees_preserve_nonnegative_integers():
    cfg = config.load_config(
        HERE / "config.yaml", ["subtract.basis_degrees=[0, 1, 2]"]
    )
    assert cfg["subtract"]["basis_degrees"] == [0, 1, 2]


@pytest.mark.parametrize("value", [".nan", ".inf", "-.inf"])
def test_numeric_settings_reject_nonfinite_values(value):
    with pytest.raises(ValueError, match="decision_threshold must be finite"):
        config.load_config(
            HERE / "config.yaml", [f"classify.decision_threshold={value}"]
        )


def test_noise_matching_requires_positive_reference_sigma():
    with pytest.raises(ValueError, match="noise_reference_sigma must be > 0"):
        config.load_config(
            HERE / "config.yaml", ["classify.noise_reference_sigma=0"]
        )
    cfg = config.load_config(
        HERE / "config.yaml",
        [
            "classify.noise_reference_sigma=0",
            "classify.apply_noise_match=false",
        ],
    )
    assert cfg["classify"]["noise_reference_sigma"] == 0


@pytest.mark.parametrize(
    "value", ["0", "-1", ".nan", ".inf", "-.inf", str(10**400)]
)
def test_psf_width_requires_a_finite_positive_value(value):
    with pytest.raises(ValueError, match="psf_sigma_px must be finite"):
        config.load_config(
            HERE / "config.yaml", [f"xfit.psf_sigma_px={value}"]
        )


@pytest.mark.parametrize("value, expected", [("null", None), ("1.5", 1.5)])
def test_psf_width_accepts_an_independent_width_or_no_photometry(
    value, expected
):
    cfg = config.load_config(
        HERE / "config.yaml", [f"xfit.psf_sigma_px={value}"]
    )
    assert cfg["xfit"]["psf_sigma_px"] == expected


def test_uq_block_defaults_and_sky_term_rule():
    raw = yaml.safe_load((HERE / "config.yaml").read_text())
    cfg = config.validate_config(raw)
    assert cfg["uq"] == {
        "enabled": True,
        "kernel_fit": True,
        "max_lobe_reach_px": 32,
        "measure_correlation": True,
        "correlation_positions": 1000,
        "sky_astrometry_arcsec": None,
        "sky_astrometry_source": None,
    }
    good = copy.deepcopy(raw)
    good["uq"] = {
        "sky_astrometry_arcsec": 0.01,
        "sky_astrometry_source": "survey astrometric summary",
    }
    assert config.validate_config(good)["uq"]["sky_astrometry_arcsec"] == 0.01
    for block in (
        {"sky_astrometry_arcsec": 0.01},
        {"sky_astrometry_source": "survey"},
    ):
        bad = copy.deepcopy(raw)
        bad["uq"] = block
        with pytest.raises(ValueError, match="set together"):
            config.validate_config(bad)
    bad = copy.deepcopy(raw)
    bad["uq"] = {"sky_astrometry_arcsec": -1.0, "sky_astrometry_source": "s"}
    with pytest.raises(ValueError, match=">= 0"):
        config.validate_config(bad)
    bad = copy.deepcopy(raw)
    bad["uq"] = {"enabled": "yes"}
    with pytest.raises(ValueError, match="true or false"):
        config.validate_config(bad)
    bad = copy.deepcopy(raw)
    bad["uq"] = {"sky_astrometry_arcsec": 0.01, "sky_astrometry_source": " "}
    with pytest.raises(ValueError, match="non-empty string"):
        config.validate_config(bad)


@pytest.mark.parametrize("value", [0, -1, True, float("nan"), 10**400])
def test_lobe_reach_limit_requires_a_positive_finite_number(value):
    raw = yaml.safe_load((HERE / "config.yaml").read_text())
    raw["uq"]["max_lobe_reach_px"] = value
    with pytest.raises(ValueError, match="max_lobe_reach_px"):
        config.validate_config(raw)


def test_external_astrometric_uncertainty_rejects_overflow():
    with pytest.raises(
        ValueError, match="sky_astrometry_arcsec must be finite"
    ):
        config.load_config(
            HERE / "config.yaml", [f"uq.sky_astrometry_arcsec={10**400}"]
        )
