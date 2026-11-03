/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "imaging_api.h"

namespace {
using namespace cuphoton::imaging;
PyMethodDef methods[] = {
    {"copy_array",
        copy_array,
        METH_VARARGS,
        "Copy or convert a strided CUDA array."},
    {"gather_bytes",
        gather_bytes,
        METH_VARARGS,
        "Gather selected bytes from a device buffer."},
    {"align_deflate",
        align_deflate,
        METH_VARARGS,
        "Pack a batch of DEFLATE payloads at aligned offsets."},
    {"validate_mask",
        validate_mask,
        METH_VARARGS,
        "Validate a binary numeric mask."},
    {"selection_image",
        selection_image,
        METH_VARARGS,
        "Prepare the finite mean image for source selection."},
    {"selection_candidates",
        selection_candidates,
        METH_VARARGS,
        "Filter and score compact sources on the GPU."},
    {"stamp_mask",
        stamp_mask,
        METH_VARARGS,
        "Rasterize selected source stamps."},
    {"prepare_mapping",
        prepare_mapping,
        METH_VARARGS,
        "Expand a coarse WCS mapping on the GPU."},
    {"reproject",
        reproject,
        METH_VARARGS,
        "Resample image, variance and mask planes."},
    {"mapping_edge_score",
        mapping_edge_score,
        METH_VARARGS,
        "Measure the source-edge distance of a mapping."},
    {"sample_mapping",
        sample_mapping,
        METH_VARARGS,
        "Gather selected mapping coordinates."},
    {"solve_constant_kernel",
        solve_constant_kernel,
        METH_VARARGS,
        "Fit and apply a constant image kernel."},
    {"accumulate_spatial_qr",
        accumulate_spatial_qr,
        METH_VARARGS,
        "Accumulate a spatial image-kernel fit with FP64 block QR."},
    {"evaluate_spatial_kernel",
        evaluate_spatial_kernel,
        METH_VARARGS,
        "Apply a spatial image kernel and propagate source variance."},
    {"restore_fits_tiles",
        restore_fits_tiles,
        METH_VARARGS,
        "Restore shuffled big-endian FITS tiles."},
    {"dequantize_fits_tiles",
        dequantize_fits_tiles,
        METH_VARARGS,
        "Restore quantized FITS tile values."},
    {nullptr, nullptr, 0, nullptr},
};
PyModuleDef_Slot slots[] = {
#if PY_VERSION_HEX >= 0x030D0000
    {Py_mod_gil, Py_MOD_GIL_NOT_USED},
#endif
    {0, nullptr},
};
PyModuleDef module = {PyModuleDef_HEAD_INIT,
    "_imaging_ext",
    nullptr,
    0,
    methods,
    slots,
    nullptr,
    nullptr,
    nullptr};
}  // namespace

PyMODINIT_FUNC PyInit__imaging_ext() {
    return PyModuleDef_Init(&module);
}
