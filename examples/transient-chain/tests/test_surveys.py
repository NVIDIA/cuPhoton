# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Survey adapters and selections on the host (NumPy, Astropy reader): layout,
masks, variance, templates, identity.
"""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pytest
from transient_chain import workload
from transient_chain.fitsio import PlaneReader, inspect_plane
from transient_chain.surveys import make_adapter
from transient_chain.surveys.base import Identity, resolve_mask_bits


def host_reader(policy="astropy"):
    return PlaneReader(policy, device=False)


def test_rubin_adapter_reads_planes_and_resolves_hsc_masklite(rubin_visit):
    adapter = make_adapter({"name": "rubin", "params": {}})
    t = adapter.load_target(rubin_visit, host_reader())
    assert (
        t.shape == rubin_visit["_image"].shape and t.image.dtype == np.float32
    )
    assert t.variance is not None and float(t.variance[0, 0]) == 9.0
    sat = rubin_visit["_mask"] != 0
    assert (
        t.bad[10, 20] and not t.bad[50, 60]
    )  # SAT is in hsc-masklite, DETECTED is not
    assert t.bad.sum() == sat.sum() - 1
    ident = adapter.identity(rubin_visit)
    assert ident == Identity(
        "2024121000367",
        "R22_S00",
        "r",
        {
            "visit": 2024121000367,
            "detector": "R22_S00",
            "date": 20241210,
            "tract": 7850,
        },
    )
    assert (
        ident.item_id == "R22_S00-v2024121000367"
        and ident.matches("R22_S00")
        and ident.matches("R22_S00/2024121000367")
    )


def test_rubin_mosaic_covers_the_visit_and_is_refinable(rubin_visit):
    adapter = make_adapter({"name": "rubin"})
    io = host_reader()
    t = adapter.load_target(rubin_visit, io)
    provider = adapter.template(rubin_visit, t, io, mapping_grid_step=16)
    assert provider.refinable
    timg, tbad, cnt = provider.build()
    assert timg.shape == t.shape and tbad is not None and cnt is not None
    finite = np.isfinite(timg)
    assert (
        finite.mean() > 0.9 and int(cnt.max()) == 2
    )  # two patches overlap in the middle
    assert np.nanmedian(timg[finite] - t.image[finite]) == pytest.approx(
        0.0, abs=2.0
    )
    timg2, _, _ = provider.build((np.eye(2), np.array([0.0, 0.0])))
    assert np.allclose(np.nan_to_num(timg2), np.nan_to_num(timg))
    assert (
        all(r["reader"] == "astropy" for r in io.receipts)
        and len(io.receipts) == 3
    )  # visit + two coadds


def test_rubin_mask_policy_override_and_missing_planes(rubin_visit):
    strict = make_adapter(
        {"name": "rubin", "params": {"mask_policy": "strict"}}
    )
    t = strict.load_target(rubin_visit, host_reader())
    assert (
        t.bad.sum() == (rubin_visit["_mask"] != 0).sum()
    )  # DETECTED counts under strict
    from astropy.io import fits

    header = fits.Header()
    header["MP_BAD"] = 0
    header["MP_SAT"] = 1
    with pytest.raises(
        ValueError, match="lacks MP_ plane cards for \\['INTRP'"
    ):
        resolve_mask_bits("hsc-masklite", header)
    assert resolve_mask_bits({"planes": ["BAD", "SAT"]}, header) == 0b11
    assert (
        resolve_mask_bits({"bits": 5}, header) == 5
        and resolve_mask_bits("none", header) == 0
        and resolve_mask_bits("strict", header) is None
    )
    with pytest.raises(ValueError, match="unknown keys"):
        make_adapter({"name": "rubin", "params": {"hdu": 1}})


def test_generic_adapter_separate_files_inverse_variance_and_grid_template(
    generic_pair,
):
    params = {
        "identity": {
            "exposure_id": "expnum",
            "ccd_id": "ccdnum",
            "band": "band",
        },
        "target": {
            "path_key": "image",
            "image_hdu": 0,
            "mask_path_key": "mask",
            "mask_hdu": 0,
            "variance_path_key": "weight",
            "variance_hdu": 0,
            "variance_kind": "inverse_variance",
        },
        "template": {
            "path_key": "template",
            "image_hdu": 0,
            "on_target_grid": True,
        },
        "mask_policy": "strict",
    }
    adapter = make_adapter({"name": "generic", "params": params})
    assert set(adapter.required_keys) == {
        "expnum",
        "ccdnum",
        "band",
        "image",
        "template",
        "mask",
        "weight",
    }
    io = host_reader()
    t = adapter.load_target(generic_pair, io)
    assert t.bad.sum() == (generic_pair["_mask"] != 0).sum()
    assert float(t.variance[5, 5]) == pytest.approx(25.0) and np.isnan(
        t.variance[0, 0]
    )  # zero weight -> NaN variance
    provider = adapter.template(generic_pair, t, io, mapping_grid_step=16)
    assert not provider.refinable
    timg, tbad, cnt = provider.build()
    assert timg.shape == t.shape and tbad is None and cnt is None
    assert adapter.identity(generic_pair).item_id == "N4-v123456"


def test_generic_adapter_defaults_need_no_mask_or_variance(generic_pair):
    adapter = make_adapter({"name": "generic"})
    t = adapter.load_target(
        {
            "exposure_id": 1,
            "ccd_id": "a",
            "band": "g",
            "image": generic_pair["image"],
            "template": generic_pair["template"],
        },
        host_reader(),
    )
    assert t.bad is None and t.variance is None and t.mask_bits is None


def test_hdu_selection_by_extname_and_first_image(rubin_visit):
    from transient_chain.fitsio import hdu_index

    assert hdu_index(rubin_visit["visit_image"], "MASK") == 2
    assert hdu_index(rubin_visit["visit_image"], None) == 1  # first 2D image
    assert (
        inspect_plane(rubin_visit["visit_image"], 1).compression == "GZIP_2"
    )
    with pytest.raises(ValueError, match="no HDU named"):
        hdu_index(rubin_visit["visit_image"], "WEIGHT")


def test_reader_policies_on_host(rubin_visit):
    arrays, receipt = PlaneReader("auto", device=False).read(
        rubin_visit["visit_image"], [1, 2]
    )
    assert (
        receipt["reader"] == "astropy"
        and receipt["requested_reader"] == "auto"
    )  # no GPU here: auto falls back
    assert arrays[1].dtype.kind in "iu"
    batch, receipts = PlaneReader("astropy", device=False).read_batch(
        rubin_visit["template_coadds"], [1, 2]
    )
    assert (
        len(batch) == 2 and receipts[0]["hdus"][0]["compression"] == "GZIP_2"
    )
    with pytest.raises(ValueError, match="xDR"):
        PlaneReader("xdr", device=False).read(rubin_visit["visit_image"], [1])


def test_selections_validation_names_the_entry_and_the_missing_file(
    tmp_path, rubin_visit
):
    adapter = make_adapter({"name": "rubin"})
    good = {k: v for k, v in rubin_visit.items() if not k.startswith("_")}
    broken = dict(good)
    broken["visit_image"] = str(tmp_path / "nope.fits")
    missing_key = {k: v for k, v in good.items() if k != "template_coadds"}
    path = tmp_path / "sel.json"
    path.write_text(json.dumps({"selections": [good, broken, missing_key]}))
    with pytest.raises(ValueError) as exc:
        workload.load_selections(path, adapter)
    text = str(exc.value)
    assert (
        "2 invalid selection(s)" in text
        and "selection 1 (R22_S00-v2024121000367): 1 input file(s) not found"
        in text
    )
    assert "selection 2 needs 'template_coadds'" in text
    path.write_text(json.dumps([good]))
    with pytest.raises(ValueError, match="bare list"):
        workload.load_selections(path, adapter)
    path.write_text(json.dumps({"shards": [good]}))
    sels = workload.load_selections(
        path, adapter, only=["R22_S00/2024121000367"]
    )
    assert len(sels) == 1
    with pytest.raises(ValueError, match="matched nothing"):
        workload.load_selections(path, adapter, only=["R22_S01"])
    items = workload.build_items(sels, adapter)
    assert (
        items[0].item_id == "R22_S00-v2024121000367"
        and items[0].weight_bytes > 0
    )
    assert set(items[0].payload) >= {
        "visit",
        "detector",
        "band",
        "visit_image",
        "template_coadds",
        "date",
        "tract",
    }


def test_generic_mask_path_without_hdu_reads_its_first_image(generic_pair):
    adapter = make_adapter(
        {
            "name": "generic",
            "params": {
                "identity": {
                    "exposure_id": "expnum",
                    "ccd_id": "ccdnum",
                    "band": "band",
                },
                "target": {"mask_path_key": "mask"},
                "mask_policy": "strict",
            },
        }
    )
    t = adapter.load_target(generic_pair, host_reader())
    assert (
        t.bad is not None
        and t.bad.sum() == (generic_pair["_mask"] != 0).sum()
    )


def test_workload_sizes_home_relative_paths():
    adapter = make_adapter({"name": "rubin"})
    with TemporaryDirectory(dir=Path.home()) as directory:
        image = Path(directory) / "visit.fits"
        image.write_bytes(b"visit")
        template = Path(directory) / "template.fits"
        template.write_bytes(b"template")
        selection = {
            "visit": 1,
            "detector": "R22_S00",
            "band": "r",
            "visit_image": "~/" + str(image.relative_to(Path.home())),
            "template_fits": "~/" + str(template.relative_to(Path.home())),
        }
        manifest = Path(directory) / "selection.json"
        manifest.write_text(json.dumps({"selections": [selection]}))
        selections = workload.load_selections(manifest, adapter)
        items = workload.build_items(selections, adapter)
        assert items[0].weight_bytes == len(b"visittemplate")
