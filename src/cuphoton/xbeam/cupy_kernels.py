# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compatibility exports for lazy GPU helpers."""

from .gpu_kernels import (  # noqa: F401
    BLOCK_PARAMETERS_DTYPE,
    FRONT_BAND_PARAMETERS_DTYPE,
    apply_residual_phase,
    batched_visibility_product,
    batched_visibility_product_packs,
    batched_visibility_products,
    channelize_and_apply_residual,
    decode_2bit_band,
    front_rotate,
    front_rotate_packed_real_bands,
    prepare_block_parameters,
    prepare_visibility_index,
    visibility_index_view,
    visibility_product,
)
from .kernel_compilation import (  # noqa: F401
    warm_gpu_pipeline,
)
from .runtime_tuning import (  # noqa: F401
    cufile_high_priority,
    cufile_strip_stream,
    vdif_upload_staging_slots,
    xmac_packs_per_launch,
)
from .upload_resources import (  # noqa: F401
    VDIF_TRANSFER_CHUNK_BYTES,
    CuFileReadStatistics,
    CuFileReadStatisticsSnapshot,
    automatic_vdif_upload_chunk_bytes,
    cached_cufile_staging_bytes,
    cached_cufile_staging_covers,
    cached_vdif_upload_staging_bytes,
    cached_vdif_upload_staging_covers,
    clear_cached_cufile_workspaces,
    clear_cached_vdif_upload_workspaces,
    load_vdif_payload_plans,
    reserve_vdif_ingest,
    reusable_vdif_upload_staging_bytes,
    vdif_upload_slot_limit,
    vdif_upload_staging_bytes,
)

__all__ = [
    "BLOCK_PARAMETERS_DTYPE",
    "automatic_vdif_upload_chunk_bytes",
    "apply_residual_phase",
    "batched_visibility_product",
    "batched_visibility_product_packs",
    "batched_visibility_products",
    "cached_cufile_staging_bytes",
    "cached_cufile_staging_covers",
    "cached_vdif_upload_staging_bytes",
    "cached_vdif_upload_staging_covers",
    "channelize_and_apply_residual",
    "clear_cached_cufile_workspaces",
    "clear_cached_vdif_upload_workspaces",
    "decode_2bit_band",
    "front_rotate",
    "front_rotate_packed_real_bands",
    "load_vdif_payload_plans",
    "prepare_block_parameters",
    "reusable_vdif_upload_staging_bytes",
    "reserve_vdif_ingest",
    "vdif_upload_staging_bytes",
    "vdif_upload_staging_slots",
    "vdif_upload_slot_limit",
    "visibility_product",
]
