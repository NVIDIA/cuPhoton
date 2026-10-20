# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest

from cuphoton.xray._hdf5_input import (
    SELF_CONTAINED_ENV,
    hdf5_input_reader,
    self_contained_hdf5_input,
)
from cuphoton.xray.detector_artifacts import (
    _read_hdf5_block,
    _ThreadedHdf5BlockReader,
)
from cuphoton.xray.detector_distributed import (
    _run_shard_command,
    build_detector_artifact_distributed_plan,
    render_local_shell_script,
    render_slurm_script,
)
from cuphoton.xray.hdf5 import load_hdf5_cube_trace, probe_hdf5_file


@pytest.fixture(autouse=True)
def self_contained_policy(monkeypatch):
    monkeypatch.setenv(SELF_CONTAINED_ENV, "1")


def _write_cube(path):
    with h5py.File(path, "w") as h5:
        h5["imgs"] = np.array([[[2.0]], [[6.0]]])
        h5["scan_var"] = [0.0, 1.0]
        h5["i0"] = [1.0, 2.0]
        h5["i0_ipm3"] = [1.0, 1.0]
        h5["bin_count"] = [1.0, 1.0]
        h5["ROI"] = [0, 0, 1, 1]


def _replace_with_external_dataset(h5, name, mechanism, missing_path):
    shape = h5[name].shape
    del h5[name]
    if mechanism == "external links":
        h5[name] = h5py.ExternalLink(str(missing_path), "/values")
    elif mechanism == "external raw storage":
        h5.create_dataset(
            name,
            shape=shape,
            dtype="f8",
            external=[(str(missing_path), 0, int(np.prod(shape)) * 8)],
        )
    else:
        assert mechanism == "virtual datasets"
        layout = h5py.VirtualLayout(shape=shape, dtype="f8")
        layout[:] = h5py.VirtualSource(
            str(missing_path), "/values", shape=shape
        )
        h5.create_virtual_dataset(name, layout)


@pytest.mark.parametrize("hard_alias", [False, True])
def test_in_file_cube_and_hard_alias_preserve_trace(tmp_path, hard_alias):
    path = tmp_path / "cube.h5"
    _write_cube(path)
    with h5py.File(path, "a") as h5:
        if hard_alias:
            h5.create_group("storage")
            h5.move("imgs", "storage/pixels")
            h5["imgs"] = h5["storage/pixels"]
            np.testing.assert_array_equal(
                hdf5_input_reader(h5)("/storage/pixels")[:], h5["imgs"][:]
            )
        metadata = h5.create_group("metadata")
        metadata["self"] = metadata
        metadata["unused"] = h5py.ExternalLink(
            str(tmp_path / "missing-metadata.h5"), "/unused"
        )

    probe = probe_hdf5_file(path)
    assert probe.schema == "cropped-cube"
    trace = load_hdf5_cube_trace(path, drop_leading=0)
    np.testing.assert_array_equal(trace.normalized_sum, [2.0, 3.0])


@pytest.mark.parametrize(
    ("link_kind", "message"),
    [
        ("external", "external links"),
        ("dangling-soft", "soft links"),
        ("internal-soft", "soft links"),
        ("soft-to-external", "soft links"),
    ],
)
def test_reject_links_before_target_access(tmp_path, link_kind, message):
    path = tmp_path / "links.h5"
    with h5py.File(path, "w") as h5:
        if link_kind == "external":
            h5["pixels"] = h5py.ExternalLink(
                str(tmp_path / "missing.h5"), "/pixels"
            )
        elif link_kind == "dangling-soft":
            h5["pixels"] = h5py.SoftLink("/missing")
        elif link_kind == "internal-soft":
            h5["values"] = [1.0, 2.0]
            h5["pixels"] = h5py.SoftLink("/values")
        else:
            h5["outside"] = h5py.ExternalLink(
                str(tmp_path / "missing.h5"), "/pixels"
            )
            h5["pixels"] = h5py.SoftLink("/outside")

    with h5py.File(path, "r") as h5:
        with pytest.raises(ValueError, match=f"HDF5 input.*{message}"):
            hdf5_input_reader(h5)("pixels")


@pytest.mark.parametrize("link_kind", ["external", "soft"])
def test_reject_intermediate_link_before_target_access(tmp_path, link_kind):
    path = tmp_path / "groups.h5"
    with h5py.File(path, "w") as h5:
        h5["outside"] = (
            h5py.ExternalLink(str(tmp_path / "missing.h5"), "/group")
            if link_kind == "external"
            else h5py.SoftLink("/missing")
        )

    with h5py.File(path, "r") as h5:
        with pytest.raises(
            ValueError, match=f"HDF5 input.*{link_kind} links"
        ):
            hdf5_input_reader(h5)("/outside/pixels")


@pytest.mark.parametrize("unlimited", [False, True])
def test_reject_same_file_virtual_dataset(tmp_path, unlimited):
    path = tmp_path / "virtual.h5"
    maxshape = (None,) if unlimited else (2,)
    with h5py.File(path, "w") as h5:
        source = h5.create_dataset(
            "values", data=[1.0, 2.0], maxshape=maxshape
        )
        layout = h5py.VirtualLayout(shape=(2,), maxshape=maxshape, dtype="f8")
        selection = slice(None, h5py.h5s.UNLIMITED if unlimited else None)
        layout[selection] = h5py.VirtualSource(source)[selection]
        h5.create_virtual_dataset("pixels", layout)

    with h5py.File(path, "r") as h5:
        with pytest.raises(ValueError, match="HDF5 input.*virtual datasets"):
            hdf5_input_reader(h5)("pixels")


def test_probe_rejects_unused_root_link(tmp_path):
    path = tmp_path / "cube.h5"
    _write_cube(path)
    with h5py.File(path, "a") as h5:
        h5["unused"] = h5py.ExternalLink(
            str(tmp_path / "missing-metadata.h5"), "/unused"
        )

    with pytest.raises(ValueError, match="HDF5 input.*external links"):
        probe_hdf5_file(path)


@pytest.mark.parametrize(
    ("name", "mechanism"),
    [
        ("imgs", "external links"),
        ("scan_var", "external raw storage"),
        ("i0", "virtual datasets"),
    ],
)
def test_probe_and_trace_reject_referenced_data(tmp_path, name, mechanism):
    path = tmp_path / "cube.h5"
    missing_path = tmp_path / "missing-source"
    _write_cube(path)
    with h5py.File(path, "a") as h5:
        _replace_with_external_dataset(h5, name, mechanism, missing_path)

    with pytest.raises(ValueError, match=f"HDF5 input.*{mechanism}"):
        probe_hdf5_file(path)
    with pytest.raises(ValueError, match=f"HDF5 input.*{mechanism}"):
        load_hdf5_cube_trace(path, drop_leading=0)


def test_block_reread_checks_replaced_dataset_after_safe_probe(tmp_path):
    path = tmp_path / "cube.h5"
    _write_cube(path)
    probe = probe_hdf5_file(path)
    assert probe.load_plan is not None
    dataset = probe.load_plan.image_dataset
    np.testing.assert_array_equal(
        _read_hdf5_block(path, dataset, np.s_[:]), [[[2.0]], [[6.0]]]
    )

    with h5py.File(path, "a") as h5:
        _replace_with_external_dataset(
            h5, dataset, "external raw storage", tmp_path / "missing-raw"
        )

    with pytest.raises(ValueError, match="HDF5 input.*external raw storage"):
        _read_hdf5_block(path, dataset, np.s_[:])


@pytest.mark.parametrize("setting", [None, "0"])
@pytest.mark.parametrize("storage", ["soft", "external", "raw", "virtual"])
def test_default_policy_preserves_linked_cube_traces(
    tmp_path, monkeypatch, setting, storage
):
    if setting is None:
        monkeypatch.delenv(SELF_CONTAINED_ENV)
    else:
        monkeypatch.setenv(SELF_CONTAINED_ENV, setting)
    path = tmp_path / "cube.h5"
    source_path = tmp_path / "source.h5"
    _write_cube(path)
    _write_cube(source_path)
    with h5py.File(path, "a") as h5:
        values = h5["imgs"][:]
        del h5["imgs"]
        if storage == "soft":
            h5["pixels"] = values
            h5["imgs"] = h5py.SoftLink("/pixels")
        elif storage == "external":
            h5["imgs"] = h5py.ExternalLink(str(source_path), "/imgs")
        elif storage == "raw":
            raw_path = tmp_path / "pixels.bin"
            raw_path.write_bytes(values.tobytes())
            h5.create_dataset(
                "imgs",
                shape=values.shape,
                dtype=values.dtype,
                external=[(str(raw_path), 0, values.nbytes)],
            )
        else:
            layout = h5py.VirtualLayout(shape=values.shape, dtype="f8")
            layout[:] = h5py.VirtualSource(
                str(source_path), "/imgs", shape=values.shape
            )
            h5.create_virtual_dataset("imgs", layout)

    trace = load_hdf5_cube_trace(path, drop_leading=0)
    np.testing.assert_array_equal(trace.normalized_sum, [2.0, 3.0])


def test_invalid_policy_reports_setting(monkeypatch):
    monkeypatch.setenv(SELF_CONTAINED_ENV, "typo")
    with pytest.raises(ValueError, match=f"{SELF_CONTAINED_ENV} must be"):
        self_contained_hdf5_input()


@pytest.mark.parametrize("enabled", [False, True])
def test_new_handle_reader_observes_current_policy(
    tmp_path, monkeypatch, enabled
):
    path = tmp_path / "linked.h5"
    with h5py.File(path, "w") as h5:
        h5["values"] = [1.0, 2.0]
        h5["pixels"] = h5py.SoftLink("/values")

    monkeypatch.setenv(SELF_CONTAINED_ENV, str(int(enabled)))
    with h5py.File(path, "r") as h5:
        read_input = hdf5_input_reader(h5)
        monkeypatch.setenv(SELF_CONTAINED_ENV, str(int(not enabled)))
        if enabled:
            with pytest.raises(ValueError, match="soft links"):
                read_input("pixels")
        else:
            np.testing.assert_array_equal(read_input("pixels")[:], [1, 2])

    with h5py.File(path, "r") as h5:
        read_input = hdf5_input_reader(h5)
        if enabled:
            np.testing.assert_array_equal(read_input("pixels")[:], [1, 2])
        else:
            with pytest.raises(ValueError, match="soft links"):
                read_input("pixels")


@pytest.mark.parametrize("enabled", [False, True])
def test_threaded_reader_retains_policy_and_rechecks_reopened_handles(
    tmp_path, monkeypatch, enabled
):
    on_path, off_path = tmp_path / "on.h5", tmp_path / "off.h5"
    _write_cube(on_path)
    _write_cube(off_path)
    monkeypatch.setenv(SELF_CONTAINED_ENV, str(int(enabled)))
    with _ThreadedHdf5BlockReader(
        on_path=on_path,
        off_path=off_path,
        image_dataset="imgs",
        workers=2,
    ) as reader:
        before = reader.read_pair(np.s_[:])
        monkeypatch.setenv(SELF_CONTAINED_ENV, str(int(not enabled)))
        with h5py.File(on_path, "a") as h5:
            h5.move("imgs", "values")
            h5["imgs"] = h5py.SoftLink("/values")
        if enabled:
            with pytest.raises(RuntimeError, match="on HDF5") as error:
                reader.read_pair(np.s_[:])
            assert isinstance(error.value.__cause__, ValueError)
            assert "soft links" in str(error.value.__cause__)
        else:
            after = reader.read_pair(np.s_[:])
            for old, new in zip(before, after, strict=True):
                np.testing.assert_array_equal(old, new)


@pytest.mark.parametrize("launcher", ["process", "local", "slurm"])
def test_saved_distributed_commands_retain_policy(
    tmp_path, monkeypatch, launcher
):
    _write_cube(tmp_path / "on.h5")
    _write_cube(tmp_path / "off.h5")
    plan = build_detector_artifact_distributed_plan(
        h5dir=tmp_path,
        fon="on.h5",
        foff="off.h5",
        output_dir=tmp_path / "out",
        tile_shape=(1, 1),
    )
    # Run the CPU probe through the generated worker launcher so this test
    # exercises the process boundary without requiring a GPU allocation.
    command = plan["commands"][0]
    action = command.index("detector-artifacts")
    plan["commands"] = [
        [
            *command[:action],
            "data-probe",
            "--h5dir",
            str(tmp_path),
            "--fon",
            "on.h5",
            "--foff",
            "off.h5",
        ]
    ]
    with h5py.File(tmp_path / "on.h5", "a") as h5:
        _replace_with_external_dataset(
            h5, "imgs", "external raw storage", tmp_path / "missing-raw"
        )
    monkeypatch.delenv(SELF_CONTAINED_ENV)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    if launcher == "process":
        result = _run_shard_command(
            plan["commands"][0],
            plan["shards"][0],
            device="0",
            work_dir=Path(plan["work_dir"]),
            resume=False,
        )
        returncode = result["returncode"]
        stderr = Path(result["log_path"]).read_text()
    else:
        script = (
            render_local_shell_script(plan, merge=False, resume=False)
            if launcher == "local"
            else render_slurm_script(plan, {})
        )
        proc = subprocess.run(
            ["bash", "-c", script],
            text=True,
            capture_output=True,
            check=False,
        )
        returncode = proc.returncode
        stderr = (
            Path(plan["shards"][0]["log_path"]).read_text()
            if launcher == "local"
            else proc.stderr
        )
    assert returncode != 0
    assert "HDF5 input 'imgs' does not support external raw storage" in stderr
