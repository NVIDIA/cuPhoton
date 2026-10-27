# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from cuphoton.ximage.backends import torch_backend
from cuphoton.ximage.backends.torch_backend import TorchBackend
from cuphoton.ximage.config import PsfSolveConfig, ReconstructionConfig
from cuphoton.ximage.data import XImageBundle

torch = pytest.importorskip("torch")


@pytest.mark.parametrize("benchmark_limit", [None, 0, 2])
def test_device_summary_preserves_unavailable_cudnn_benchmark_limit(
    benchmark_limit: int | None,
) -> None:
    cpu_torch = SimpleNamespace(
        __version__="cpu-test",
        version=SimpleNamespace(cuda=None),
        cuda=SimpleNamespace(is_available=lambda: False),
        backends=SimpleNamespace(
            cudnn=SimpleNamespace(
                version=lambda: None,
                benchmark=False,
                benchmark_limit=benchmark_limit,
                deterministic=False,
            )
        ),
    )
    summary = torch_backend._torch_device_summary(
        cpu_torch, torch.device("cpu")
    )
    assert summary["cudnn_benchmark_limit"] == benchmark_limit
    assert summary["cudnn_version"] is None
    assert summary["cuda_available"] is False


def _bundle(psfs: np.ndarray) -> XImageBundle:
    images = np.zeros((1, 8, 8, psfs.shape[3]), dtype=np.float64)
    images[:, 3:5, 3:5, :] = 1.0
    masks = np.ones((1, 8, 8, 12, psfs.shape[3]), dtype=np.uint8)
    sky = np.ones((1, 8, 8, psfs.shape[3]), dtype=np.uint8)
    variances = np.ones((1, 8, 8, psfs.shape[3]), dtype=np.float64)
    exp_times = np.full((psfs.shape[3],), 300.0, dtype=np.float64)
    mcgauss = np.zeros((psfs.shape[3], 3, 3), dtype=np.float64)
    return XImageBundle(
        store_format="npy",
        store_path=Path("."),
        arrays={
            "images": images,
            "masks": masks,
            "sky": sky,
            "variances": variances,
            "psfs": psfs,
            "exp_times": exp_times,
            "mcgauss": mcgauss,
        },
        metadata={},
    )


def _outlier_bundle(
    psfs: np.ndarray,
    *,
    outlier: float = 100.0,
    variance: float = 1.0,
) -> XImageBundle:
    images = np.ones((1, 8, 8, psfs.shape[3]), dtype=np.float64)
    images[:, 4, 4, 0] = outlier
    if psfs.shape[3] > 1:
        images[:, 4, 4, 1:] = 1.0
    masks = np.ones((1, 8, 8, 12, psfs.shape[3]), dtype=np.uint8)
    sky = np.ones((1, 8, 8, psfs.shape[3]), dtype=np.uint8)
    variances = np.full(
        (1, 8, 8, psfs.shape[3]),
        variance,
        dtype=np.float64,
    )
    exp_times = np.ones((psfs.shape[3],), dtype=np.float64)
    mcgauss = np.zeros((psfs.shape[3], 3, 3), dtype=np.float64)
    return XImageBundle(
        store_format="npy",
        store_path=Path("."),
        arrays={
            "images": images,
            "masks": masks,
            "sky": sky,
            "variances": variances,
            "psfs": psfs,
            "exp_times": exp_times,
            "mcgauss": mcgauss,
        },
        metadata={},
    )


def _offset_bundle(psfs: np.ndarray, offsets: np.ndarray) -> XImageBundle:
    images = np.zeros((1, 8, 8, psfs.shape[3]), dtype=np.float64)
    images[:, 3:5, 3:5, :] = 1.0
    images = images + offsets.reshape(1, 1, 1, -1)
    masks = np.ones((1, 8, 8, 12, psfs.shape[3]), dtype=np.uint8)
    sky = np.ones((1, 8, 8, psfs.shape[3]), dtype=np.uint8)
    variances = np.ones((1, 8, 8, psfs.shape[3]), dtype=np.float64)
    exp_times = np.ones((psfs.shape[3],), dtype=np.float64)
    mcgauss = np.zeros((psfs.shape[3], 3, 3), dtype=np.float64)
    return XImageBundle(
        store_format="npy",
        store_path=Path("."),
        arrays={
            "images": images,
            "masks": masks,
            "sky": sky,
            "variances": variances,
            "psfs": psfs,
            "exp_times": exp_times,
            "mcgauss": mcgauss,
        },
        metadata={},
    )


def _config() -> PsfSolveConfig:
    return PsfSolveConfig.model_validate(
        {
            "backend": "torch",
            "device": "cpu",
            "method": "adam",
            "data": {"base": "data"},
            "output": {
                "root_dir": "runs",
                "name": "torch-psf-test",
                "overwrite": True,
            },
            "psf_artifact": "psfs",
            "srf": 1,
            "sigma_g": 0.55,
            "padding": "VALID",
            "h_initialization": "ones",
            "loss": "L2",
            "grad_reg_coeff": 0.0,
            "lr": 1e-3,
            "num_steps": 2,
        }
    )


def _reconstruct_config() -> ReconstructionConfig:
    return ReconstructionConfig.model_validate(
        {
            "backend": "torch",
            "device": "cpu",
            "algorithm": "l2",
            "data": {"base": "data"},
            "output": {
                "root_dir": "runs",
                "name": "torch-reconstruct-test",
                "overwrite": True,
            },
            "roi": {"center": [4, 4], "height": 4, "width": 4},
            "mask": {"mode": "none", "dilation_factor": 1},
            "srf": 1,
            "x_initialization": "median",
            "ucf": 2.0,
            "learning_rate": 1.0,
            "max_iter": 1,
            "stop_tol": 1e-6,
            "effective_mask_mu": 0.1,
            "exposure_batch_size": None,
            "padding": "SAME",
            "verbose": False,
        }
    )


def test_torch_backend_reports_availability() -> None:
    backend = TorchBackend()
    assert backend.is_available() is True


def test_torch_psf_solve_returns_expected_shape() -> None:
    backend = TorchBackend()
    psfs = np.zeros((5, 5, 1, 3), dtype=np.float64)
    psfs[2, 2, 0, :] = 1.0
    result = backend.solve_psf(_bundle(psfs), _config())
    assert result["psfs"].shape == psfs.shape
    assert result["gaussian_psf"].shape[0] == 1
    assert result["trace"]["loss"].shape[0] == 2


def test_torch_psf_solve_supports_super_resolution_shape() -> None:
    backend = TorchBackend()
    psfs = np.zeros((5, 5, 1, 2), dtype=np.float64)
    psfs[2, 2, 0, :] = 1.0
    config = _config().model_copy(update={"srf": 2})
    result = backend.solve_psf(_bundle(psfs), config)
    assert result["psfs"].shape == (9, 9, 1, 2)


def test_torch_psf_solve_rejects_even_target_shape() -> None:
    backend = TorchBackend()
    psfs = np.zeros((4, 4, 1, 1), dtype=np.float64)

    with pytest.raises(ValueError, match="square with odd"):
        backend.solve_psf(_bundle(psfs), _config())


def test_torch_psf_solve_supports_grad_regularization() -> None:
    backend = TorchBackend()
    psfs = np.zeros((5, 5, 1, 1), dtype=np.float64)
    psfs[2, 2, 0, 0] = 1.0
    config = _config().model_copy(
        update={"grad_reg_coeff": 1e-3, "num_steps": 1}
    )
    result = backend.solve_psf(_bundle(psfs), config)
    assert result["trace"]["loss"].shape[0] == 1


def test_torch_psf_solve_supports_mm() -> None:
    backend = TorchBackend()
    config = _config().model_copy(
        update={"method": "mm", "min_iter": 1, "max_iter": 3}
    )
    psfs = np.zeros((5, 5, 1, 1), dtype=np.float64)
    psfs[2, 2, 0, 0] = 1.0

    result = backend.solve_psf(_bundle(psfs), config)

    assert result["psfs"].shape == psfs.shape
    assert result["padded_psfs"].shape == (9, 9, 1, 1)
    assert result["trace"][0]["loss"].shape == (3,)
    assert result["trace"][0]["loss"][-1] < result["trace"][0]["loss"][0]


def test_torch_psf_solve_random_initialization_is_seeded() -> None:
    backend = TorchBackend()
    psfs = np.zeros((5, 5, 1, 1), dtype=np.float64)
    psfs[2, 2, 0, 0] = 1.0
    config = _config().model_copy(
        update={"h_initialization": "random", "seed": 17, "num_steps": 1}
    )

    first = backend.solve_psf(_bundle(psfs), config)
    second = backend.solve_psf(_bundle(psfs), config)

    np.testing.assert_array_equal(first["psfs"], second["psfs"])


def test_torch_psf_solve_mm_rejects_zero_initialization() -> None:
    backend = TorchBackend()
    psfs = np.zeros((5, 5, 1, 1), dtype=np.float64)
    config = _config().model_copy(
        update={"method": "mm", "h_initialization": "zeros"}
    )

    with pytest.raises(ValueError, match="multiplicative updates"):
        backend.solve_psf(_bundle(psfs), config)


@pytest.mark.parametrize("algorithm", ["l2", "rho"])
def test_reconstruction_copies_only_final_image_and_retains_scalar_traces(
    algorithm, monkeypatch
) -> None:
    copied_shapes = []
    original = torch_backend._to_numpy_nhwc

    def record_copy(tensor):
        copied_shapes.append(tuple(tensor.shape))
        return original(tensor)

    monkeypatch.setattr(torch_backend, "_to_numpy_nhwc", record_copy)
    backend = TorchBackend()
    psfs = np.zeros((3, 3, 1, 1), dtype=np.float64)
    psfs[1, 1, 0, 0] = 1.0
    config = _reconstruct_config().model_copy(
        update={"algorithm": algorithm, "max_iter": 3}
    )
    result = backend.reconstruct(_bundle(psfs), config)
    assert result["image"].shape == (1, 4, 4, 1)
    assert copied_shapes == [(1, 1, 4, 4)]
    assert set(result["trace"]) == {"iterations", "loss", "u_score"}
    np.testing.assert_array_equal(result["trace"]["iterations"], [0, 1, 2])
    assert result["trace"]["loss"].shape == (3,)
    assert result["trace"]["u_score"].shape == (1,)


def test_torch_rho_scalar_background_records_per_exposure_offsets() -> None:
    backend = TorchBackend()
    psfs = np.zeros((3, 3, 1, 2), dtype=np.float64)
    psfs[1, 1, 0, :] = 1.0
    config = _reconstruct_config().model_copy(
        update={
            "algorithm": "rho",
            "background_model": "scalar_per_exposure",
            "max_iter": 1,
            "delta_rho_start": 10.0,
            "delta_rho_end": 10.0,
        }
    )

    result = backend.reconstruct(
        _offset_bundle(psfs, np.asarray([0.0, 10.0])),
        config,
    )

    background = result["trace"]["background"]
    assert background.shape == (1, 1, 2)
    np.testing.assert_allclose(background[0, 0], [-5.0, 5.0], atol=1e-12)


def test_torch_rho_downweights_normalized_outliers() -> None:
    backend = TorchBackend()
    psfs = np.zeros((3, 3, 1, 3), dtype=np.float64)
    psfs[1, 1, 0, :] = 1.0
    config = _reconstruct_config().model_copy(
        update={
            "algorithm": "rho",
            "max_iter": 1,
            "delta_rho_start": 1.0,
            "delta_rho_end": 1.0,
        }
    )
    result = backend.reconstruct(_outlier_bundle(psfs), config)

    # Residuals [99, 0, 0] give Huber weights [1/99, 1, 1].
    np.testing.assert_allclose(result["image"][0, 2, 2, 0], 298.0 / 199.0)


def test_torch_rho_huber_threshold_uses_normalized_residual_units() -> None:
    backend = TorchBackend()
    psfs = np.zeros((3, 3, 1, 3), dtype=np.float64)
    psfs[1, 1, 0, :] = 1.0
    config = _reconstruct_config().model_copy(
        update={
            "algorithm": "rho",
            "max_iter": 1,
            "ucf": 20.0,
            "delta_rho_start": 1.0,
            "delta_rho_end": 1.0,
        }
    )
    result = backend.reconstruct(
        _outlier_bundle(psfs, outlier=101.0, variance=100.0),
        config,
    )

    # Variance 100 gives normalized residuals [10, 0, 0], hence a 1:10
    # outlier-to-inlier weight ratio and the weighted mean 121/21.
    np.testing.assert_allclose(result["image"][0, 2, 2, 0], 121.0 / 21.0)


@pytest.mark.parametrize(
    "rho_loss",
    [
        "huber",
        "cauchy",
        "bisquare",
        "damped_rl",
    ],
)
def test_torch_rho_supports_robust_losses(rho_loss: str) -> None:
    backend = TorchBackend()
    psfs = np.zeros((3, 3, 1, 1), dtype=np.float64)
    psfs[1, 1, 0, 0] = 1.0
    config = _reconstruct_config().model_copy(
        update={"algorithm": "rho", "rho_loss": rho_loss}
    )

    result = backend.reconstruct(_bundle(psfs), config)

    assert np.isfinite(result["image"]).all()
    assert np.isfinite(result["trace"]["loss"]).all()


def test_torch_l2_reconstruct_batching_matches_full() -> None:
    backend = TorchBackend()
    psfs = np.zeros((3, 3, 1, 3), dtype=np.float64)
    psfs[1, 1, 0, :] = 1.0
    config = _reconstruct_config().model_copy(update={"max_iter": 3})
    full = backend.reconstruct(_bundle(psfs), config)
    batched = backend.reconstruct(
        _bundle(psfs),
        config.model_copy(update={"exposure_batch_size": 1}),
    )

    np.testing.assert_allclose(batched["image"], full["image"], atol=1e-12)
    np.testing.assert_allclose(
        batched["trace"]["iterations"],
        full["trace"]["iterations"],
    )
    np.testing.assert_allclose(
        batched["trace"]["loss"], full["trace"]["loss"]
    )
    np.testing.assert_allclose(
        batched["trace"]["u_score"],
        full["trace"]["u_score"],
    )


def test_torch_l2_tiled_reconstruct_matches_full_for_dirac_psf() -> None:
    backend = TorchBackend()
    psfs = np.zeros((3, 3, 1, 3), dtype=np.float64)
    psfs[1, 1, 0, :] = 1.0
    config = _reconstruct_config().model_copy(update={"max_iter": 3})
    tiled_config = config.model_copy(
        update={
            "tiling": config.tiling.model_copy(
                update={
                    "enabled": True,
                    "tile_height": 2,
                    "tile_width": 2,
                    "halo": 2,
                }
            )
        }
    )

    full = backend.reconstruct(_bundle(psfs), config)
    tiled = backend.reconstruct(_bundle(psfs), tiled_config)

    np.testing.assert_allclose(tiled["image"], full["image"], atol=1e-12)
    assert tiled["trace"]["tiling"]["tile_count"] == 4
    assert tiled["trace"]["tiling"]["tile_height"] == 2
    assert tiled["trace"]["tiling"]["tile_width"] == 2
    assert tiled["trace"]["tiling"]["halo"] == 2
    assert tiled["trace"]["tiles"][0]["expanded_yx"] == [0, 4, 0, 4]


def test_torch_tiled_reconstruct_preserves_dtype_and_performance() -> None:
    backend = TorchBackend()
    psfs = np.zeros((3, 3, 1, 2), dtype=np.float64)
    psfs[1, 1, 0, :] = 1.0
    config = _reconstruct_config().model_copy(
        update={
            "dtype": "float32",
            "max_iter": 1,
            "tiling": _reconstruct_config().tiling.model_copy(
                update={
                    "enabled": True,
                    "tile_height": 2,
                    "tile_width": 2,
                    "halo": 2,
                }
            ),
        }
    )

    result = backend.reconstruct(_bundle(psfs), config)

    assert result["image"].dtype == np.float32
    assert result["performance"]["backend"] == "torch"
    assert result["performance"]["dtype"] == "float32"
    assert result["performance"]["device"]["device"] in {"cpu", "cuda"}
    assert result["performance"]["tile_count"] == 4
    assert result["performance"]["timings_sec"]["total"] > 0.0


def test_torch_tiled_reconstruct_rejects_valid_padding() -> None:
    backend = TorchBackend()
    psfs = np.zeros((3, 3, 1, 1), dtype=np.float64)
    psfs[1, 1, 0, :] = 1.0
    config = _reconstruct_config().model_copy(
        update={
            "padding": "VALID",
            "tiling": _reconstruct_config().tiling.model_copy(
                update={
                    "enabled": True,
                    "tile_height": 2,
                    "tile_width": 2,
                    "halo": 0,
                }
            ),
        }
    )

    with pytest.raises(NotImplementedError, match="padding='SAME'"):
        backend.reconstruct(_bundle(psfs), config)


def test_torch_rho_reconstruct_batching_matches_full() -> None:
    backend = TorchBackend()
    psfs = np.zeros((3, 3, 1, 3), dtype=np.float64)
    psfs[1, 1, 0, :] = 1.0
    config = _reconstruct_config().model_copy(
        update={"algorithm": "rho", "max_iter": 3}
    )
    full = backend.reconstruct(_bundle(psfs), config)
    batched = backend.reconstruct(
        _bundle(psfs),
        config.model_copy(update={"exposure_batch_size": 2}),
    )

    np.testing.assert_allclose(batched["image"], full["image"], atol=1e-12)
    np.testing.assert_allclose(
        batched["trace"]["loss"],
        full["trace"]["loss"],
    )
    np.testing.assert_allclose(
        batched["trace"]["u_score"],
        full["trace"]["u_score"],
    )


def test_torch_l2_batched_valid_superres_matches_full() -> None:
    backend = TorchBackend()
    psfs = np.zeros((3, 3, 1, 2), dtype=np.float64)
    psfs[1, 1, 0, :] = 1.0
    config = _reconstruct_config().model_copy(
        update={
            "padding": "VALID",
            "srf": 2,
            "max_iter": 2,
        }
    )
    full = backend.reconstruct(_bundle(psfs), config)
    batched = backend.reconstruct(
        _bundle(psfs),
        config.model_copy(update={"exposure_batch_size": 1}),
    )

    np.testing.assert_allclose(batched["image"], full["image"], atol=1e-12)
    np.testing.assert_allclose(
        batched["trace"]["loss"],
        full["trace"]["loss"],
    )


def test_torch_rho_batched_valid_superres_matches_full() -> None:
    backend = TorchBackend()
    psfs = np.zeros((3, 3, 1, 2), dtype=np.float64)
    psfs[1, 1, 0, :] = 1.0
    config = _reconstruct_config().model_copy(
        update={
            "algorithm": "rho",
            "padding": "VALID",
            "srf": 2,
            "max_iter": 2,
        }
    )
    full = backend.reconstruct(_bundle(psfs), config)
    batched = backend.reconstruct(
        _bundle(psfs),
        config.model_copy(update={"exposure_batch_size": 1}),
    )

    np.testing.assert_allclose(batched["image"], full["image"], atol=1e-12)
    np.testing.assert_allclose(
        batched["trace"]["loss"], full["trace"]["loss"]
    )


def test_torch_rho_background_batching_matches_full() -> None:
    backend = TorchBackend()
    psfs = np.zeros((3, 3, 1, 3), dtype=np.float64)
    psfs[1, 1, 0, :] = 1.0
    config = _reconstruct_config().model_copy(
        update={
            "algorithm": "rho",
            "background_model": "scalar_per_exposure",
            "max_iter": 3,
            "delta_rho_start": 10.0,
            "delta_rho_end": 10.0,
        }
    )
    bundle = _offset_bundle(psfs, np.asarray([0.0, 4.0, -2.0]))

    full = backend.reconstruct(bundle, config)
    batched = backend.reconstruct(
        bundle,
        config.model_copy(update={"exposure_batch_size": 1}),
    )

    np.testing.assert_allclose(batched["image"], full["image"], atol=1e-12)
    np.testing.assert_allclose(
        batched["trace"]["loss"], full["trace"]["loss"]
    )
    np.testing.assert_allclose(
        batched["trace"]["background"],
        full["trace"]["background"],
        atol=1e-12,
    )


@pytest.mark.parametrize("max_exposures", [None, 2, 5])
def test_external_psfs_reject_unmatched_saved_exposure_ids(
    tmp_path, max_exposures
) -> None:
    import json

    from cuphoton.ximage.backends.torch_backend import _load_psfs

    psfs = np.ones((3, 3, 1, 2), dtype=np.float64)
    path = tmp_path / "artifacts" / "psfs.npy"
    path.parent.mkdir()
    np.save(path, psfs)
    (tmp_path / "summary.json").write_text(
        json.dumps({"selected_exposures": [3, 8]})
    )
    with pytest.raises(RuntimeError, match="exposure"):
        _load_psfs(
            _bundle(psfs),
            path,
            max_exposures=max_exposures,
            target_exposure_count=2,
        )


@pytest.mark.parametrize("max_exposures", [None, 3])
@pytest.mark.parametrize("psf_count", [2, 4])
def test_external_psfs_match_selected_exposure_count(
    tmp_path, max_exposures, psf_count
) -> None:
    psfs = np.ones((3, 3, 1, psf_count), dtype=np.float64)
    path = tmp_path / "psfs.npy"
    np.save(path, psfs)
    bundle = _bundle(np.ones((3, 3, 1, 3)))
    if max_exposures == 3 and psf_count == 4:
        selected = torch_backend._load_psfs(
            bundle, path, max_exposures=3, target_exposure_count=3
        )
        np.testing.assert_array_equal(selected, psfs[..., :3])
    else:
        with pytest.raises(RuntimeError, match="selected exposure count"):
            torch_backend._load_psfs(
                bundle,
                path,
                max_exposures=max_exposures,
                target_exposure_count=3,
            )


@pytest.mark.parametrize("shape", [(3, 3), (3, 3, 1), (3, 3, 1, 1, 1)])
def test_external_psfs_reject_wrong_rank_with_clear_error(tmp_path, shape):
    path = tmp_path / "psfs.npy"
    np.save(path, np.ones(shape))
    with pytest.raises(ValueError, match="External PSFs must have shape"):
        torch_backend._load_psfs(_bundle(np.ones((3, 3, 1, 1))), path)


@pytest.mark.parametrize("shape", [(3, 3, 2, 1), (4, 4, 1, 1), (3, 5, 1, 1)])
def test_reconstruction_rejects_unsupported_psf_geometry(shape):
    psfs = np.ones(shape)
    with pytest.raises(ValueError, match="PSF"):
        TorchBackend().reconstruct(_bundle(psfs), _reconstruct_config())


def test_external_psfs_select_saved_exposure_ids_not_array_positions(
    tmp_path,
) -> None:
    import json

    from cuphoton.ximage.backends.torch_backend import _load_psfs

    psfs = np.arange(27, dtype=np.float64).reshape(3, 3, 1, 3)
    path = tmp_path / "artifacts" / "psfs.npy"
    path.parent.mkdir()
    np.save(path, psfs)
    (tmp_path / "summary.json").write_text(
        json.dumps({"selected_exposures": [3, 8, 12]})
    )
    result = _load_psfs(
        _bundle(psfs[:, :, :, :2]),
        path,
        exposure_indices=[12, 3],
        target_exposure_count=2,
    )
    np.testing.assert_array_equal(result, psfs[:, :, :, [2, 0]])


def test_external_psfs_allow_identity_prefix(tmp_path) -> None:
    import json

    from cuphoton.ximage.backends.torch_backend import _load_psfs

    psfs = np.arange(27, dtype=np.float64).reshape(3, 3, 1, 3)
    path = tmp_path / "artifacts" / "psfs.npy"
    path.parent.mkdir()
    np.save(path, psfs)
    (tmp_path / "summary.json").write_text(
        json.dumps({"selected_exposures": [0, 1, 2]})
    )
    result = _load_psfs(
        _bundle(psfs[:, :, :, :2]),
        path,
        max_exposures=2,
        target_exposure_count=2,
    )
    np.testing.assert_array_equal(result, psfs[:, :, :, :2])


@pytest.mark.gpu
@pytest.mark.parametrize("algorithm", ["l2", "rho"])
def test_cuda_reconstruction_matches_cpu_and_preserves_cudnn_settings(
    algorithm, monkeypatch
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    psfs = np.zeros((3, 3, 1, 2), dtype=np.float64)
    psfs[1, 1, 0, :] = 1.0
    bundle = _bundle(psfs)
    config = _reconstruct_config().model_copy(
        update={"algorithm": algorithm, "max_iter": 3, "stop_tol": 0.0}
    )
    backend = TorchBackend()
    expected = backend.reconstruct(bundle, config)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", False)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark_limit", 2)
    actual = backend.reconstruct(
        bundle, config.model_copy(update={"device": "cuda"})
    )
    np.testing.assert_allclose(actual["image"], expected["image"], atol=1e-12)
    np.testing.assert_allclose(
        actual["trace"]["loss"], expected["trace"]["loss"], atol=1e-12
    )
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.cudnn.benchmark_limit == 2
    assert actual["performance"]["device"]["device"] == "cuda"


def test_valid_reconstruction_rejects_full_frame_without_padding_data() -> (
    None
):
    psfs = np.zeros((3, 3, 1, 2), dtype=np.float64)
    psfs[1, 1, 0, :] = 1.0
    config = _reconstruct_config()
    config = config.model_copy(
        update={
            "padding": "VALID",
            "roi": config.roi.model_copy(update={"height": 8, "width": 8}),
        }
    )
    with pytest.raises(ValueError, match="Padded ROI exceeds.*bounds"):
        TorchBackend().reconstruct(_bundle(psfs), config)
