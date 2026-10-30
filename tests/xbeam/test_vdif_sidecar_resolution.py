# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

import hashlib
import os
import struct
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from cuphoton.xbeam.adapters.vdif import vdif_sidecar as sidecar
from cuphoton.xbeam.adapters.vdif import vdif_sidecar_builder as builder
from cuphoton.xbeam.adapters.vdif.vdif_index import VdifIndexError
from cuphoton.xbeam.adapters.vdif.vdif_sidecar_builder import (
    build_vdif_binary_index,
)


def _source(path: Path, value: int = 0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = struct.pack("<8I", 0, 40 << 24, 5, 1 << 26, 0, 0, 0, 0)
    path.write_bytes(header + bytes((value,)) * 8)
    return path


def _rewrite_document(index: Path, **changes) -> None:
    document = sidecar._read_document(index)
    document.update(changes)
    with index.open("r+b") as stream:
        stream.write(sidecar.encode_binary_index_header(document))
    sidecar._read_document_cached.cache_clear()
    sidecar._validate_cached.cache_clear()


@pytest.mark.parametrize("force", [False, True])
def test_concurrent_same_basename_builds_publish_distinct_indexes(
    tmp_path, monkeypatch, force
):
    directory = tmp_path / "indexes"
    sources = (
        _source(tmp_path / "first" / "source.vdif", 1),
        _source(tmp_path / "second" / "source.vdif", 2),
    )
    rendezvous = threading.Barrier(2)
    write_index = builder._write_cpu_index

    def synchronized_write(*args, **kwargs):
        write_index(*args, **kwargs)
        rendezvous.wait(timeout=10)

    monkeypatch.setattr(builder, "_write_cpu_index", synchronized_write)
    with ThreadPoolExecutor(max_workers=2) as executor:
        summaries = tuple(
            executor.map(
                lambda source: build_vdif_binary_index(
                    source, index_dir=directory, backend="posix", force=force
                ),
                sources,
            )
        )

    assert summaries[0].index_path != summaries[1].index_path
    assert {path.name for path in directory.glob("*.vdix")} == {
        summary.index_path.name for summary in summaries
    }
    for source, summary in zip(sources, summaries, strict=True):
        assert (
            sidecar.validate_vdif_binary_index(
                source, index_dir=directory
            ).index_path
            == summary.index_path
        )


@pytest.mark.parametrize("naming", ["basename", "collision", "path-hash"])
def test_sidecar_remains_reachable_after_parent_rename(tmp_path, naming):
    directory = tmp_path / "indexes"
    if naming == "collision":
        other = _source(tmp_path / "other" / "source.vdif", 1)
        build_vdif_binary_index(other, index_dir=directory, backend="posix")
    source = _source(tmp_path / "dataset" / "source.vdif")
    summary = build_vdif_binary_index(
        source, index_dir=directory, backend="posix"
    )
    index = summary.index_path
    if naming == "basename":
        assert index.name == "source.vdif.vdix"
    if naming == "path-hash":
        digest = hashlib.sha256(os.fsencode(str(source))).hexdigest()[:16]
        previous = directory / f"source.vdif.{digest}.vdix"
        index.rename(previous)
        index = previous
        # Legacy sidecars did not record ownership hints.
        _rewrite_document(index, source_locator=None)
    source.parent.rename(tmp_path / "moved")
    relocated = tmp_path / "moved" / source.name

    assert (
        sidecar.validate_vdif_binary_index(
            relocated, index_dir=directory
        ).index_path
        == index
    )


@pytest.mark.parametrize("collision", [False, True])
def test_stale_sidecar_requires_force_and_reuses_its_location(
    tmp_path, collision
):
    directory = tmp_path / "indexes"
    other_index = None
    if collision:
        other = _source(tmp_path / "other" / "source.vdif", 1)
        other_index = build_vdif_binary_index(
            other, index_dir=directory, backend="posix"
        ).index_path
        other_bytes = other_index.read_bytes()
    source = _source(tmp_path / "dataset" / "source.vdif")
    index = build_vdif_binary_index(
        source, index_dir=directory, backend="posix"
    ).index_path
    _source(source, 2)

    with pytest.raises(
        VdifIndexError, match="(provenance|fingerprint).+stale"
    ):
        build_vdif_binary_index(source, index_dir=directory, backend="posix")
    rebuilt = build_vdif_binary_index(
        source, index_dir=directory, backend="posix", force=True
    )
    assert rebuilt.index_path == index
    assert len(list(directory.glob("*.vdix"))) == (2 if collision else 1)
    if other_index is not None:
        assert other_index.read_bytes() == other_bytes


@pytest.mark.parametrize("corruption", ["empty", "garbage", "metadata"])
def test_corrupt_basename_sidecar_is_reported_and_force_preserves_it(
    tmp_path, corruption
):
    source = _source(tmp_path / "source.vdif")
    directory = tmp_path / "indexes"
    index = build_vdif_binary_index(
        source, index_dir=directory, backend="posix"
    ).index_path
    if corruption == "metadata":
        with index.open("r+b") as stream:
            stream.seek(16)
            stream.write(b"invalid JSON")
    else:
        index.write_bytes(b"" if corruption == "empty" else b"garbage" * 100)

    corrupted_bytes = index.read_bytes()
    with pytest.raises(VdifIndexError, match="sidecar"):
        build_vdif_binary_index(source, index_dir=directory, backend="posix")
    rebuilt = build_vdif_binary_index(
        source, index_dir=directory, backend="posix", force=True
    )
    assert rebuilt.index_path != index
    assert index.read_bytes() == corrupted_bytes
    assert set(directory.glob("*.vdix")) == {index, rebuilt.index_path}
    assert (
        sidecar.validate_vdif_binary_index(
            source, index_dir=directory
        ).index_path
        == rebuilt.index_path
    )


def test_legacy_sidecar_with_unknown_owner_does_not_imply_collision(tmp_path):
    directory = tmp_path / "indexes"
    source = _source(tmp_path / "first" / "source.vdif")
    index = build_vdif_binary_index(
        source, index_dir=directory, backend="posix"
    ).index_path
    _rewrite_document(index, source_locator=None)
    assert (
        build_vdif_binary_index(
            source, index_dir=directory, backend="posix", force=True
        ).index_path
        == index
    )
    _rewrite_document(index, source_locator=None)
    original_bytes = index.read_bytes()
    other = _source(tmp_path / "second" / source.name, 1)

    with pytest.raises(
        VdifIndexError, match="(provenance|fingerprint).+stale"
    ):
        build_vdif_binary_index(other, index_dir=directory, backend="posix")
    assert list(directory.glob("*.vdix")) == [index]
    rebuilt = build_vdif_binary_index(
        other, index_dir=directory, backend="posix", force=True
    )
    assert rebuilt.index_path != index
    assert index.read_bytes() == original_bytes
    assert (
        sidecar.validate_vdif_binary_index(
            source, index_dir=directory
        ).index_path
        == index
    )
    assert (
        sidecar.validate_vdif_binary_index(
            other, index_dir=directory
        ).index_path
        == rebuilt.index_path
    )


@pytest.mark.parametrize("relocate", [False, True])
def test_valid_legacy_hash_takes_precedence_over_unknown_basename(
    tmp_path, relocate
):
    directory = tmp_path / "indexes"
    first = _source(tmp_path / "first" / "source.vdif")
    primary = build_vdif_binary_index(
        first, index_dir=directory, backend="posix"
    ).index_path
    second = _source(tmp_path / "second" / "source.vdif", 1)
    current = build_vdif_binary_index(
        second, index_dir=directory, backend="posix"
    ).index_path
    digest = hashlib.sha256(os.fsencode(str(second))).hexdigest()[:16]
    legacy_hash = directory / f"source.vdif.{digest}.vdix"
    current.rename(legacy_hash)
    for index in (primary, legacy_hash):
        _rewrite_document(index, source_locator=None)
    primary_bytes = primary.read_bytes()
    if relocate:
        second.parent.rename(tmp_path / "moved")
        second = tmp_path / "moved" / second.name

    assert (
        sidecar.validate_vdif_binary_index(
            second, index_dir=directory
        ).index_path
        == legacy_hash
    )
    assert (
        build_vdif_binary_index(
            second, index_dir=directory, backend="posix", force=True
        ).index_path
        == legacy_hash
    )
    assert primary.read_bytes() == primary_bytes
    assert (
        sidecar.validate_vdif_binary_index(
            first, index_dir=directory
        ).index_path
        == primary
    )


@pytest.mark.parametrize("collision", [False, True])
def test_resolution_and_validation_share_one_metadata_read(
    tmp_path, monkeypatch, collision
):
    source = _source(tmp_path / "source.vdif")
    directory = tmp_path / "indexes"
    expected = []
    if collision:
        other = _source(tmp_path / "other" / "source.vdif", 1)
        expected.append(
            build_vdif_binary_index(
                other, index_dir=directory, backend="posix"
            ).index_path
        )
    index = build_vdif_binary_index(
        source, index_dir=directory, backend="posix"
    ).index_path
    expected.append(index)
    sidecar._read_document_cached.cache_clear()
    sidecar._validate_cached.cache_clear()
    original_open = Path.open
    reads = []

    def count_open(path, *args, **kwargs):
        if path.parent == directory and path.suffix == ".vdix":
            reads.append(path)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", count_open)
    assert (
        sidecar.resolve_vdif_binary_index_path(
            source, directory, frame_size_bytes=40
        )
        == index
    )
    sidecar.validate_vdif_binary_index(
        source, index_dir=directory, frame_size_bytes=40
    )
    assert reads == expected


def test_cached_metadata_is_invalidated_by_in_place_corruption(tmp_path):
    source = _source(tmp_path / "source.vdif")
    index = build_vdif_binary_index(source, backend="posix").index_path
    sidecar.validate_vdif_binary_index(source)
    index.write_bytes(b"")

    with pytest.raises(VdifIndexError, match="truncated"):
        sidecar.validate_vdif_binary_index(source)


@pytest.mark.parametrize(
    "field", ["rejected_format_frame_count", "rejected_header_frame_count"]
)
def test_prior_sidecars_with_masked_malformed_frames_require_rebuild(
    tmp_path, field
):
    source = _source(tmp_path / "source.vdif")
    index = build_vdif_binary_index(source, backend="posix").index_path
    _rewrite_document(index, **{field: 1})

    with pytest.raises(VdifIndexError, match="rebuild with --force"):
        sidecar.validate_vdif_binary_index(source)
