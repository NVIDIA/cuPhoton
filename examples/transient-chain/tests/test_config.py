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
