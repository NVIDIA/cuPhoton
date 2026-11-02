/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "imaging_api.h"
#include <array>
#include <memory>

namespace cuphoton::imaging {
namespace {
struct DecodeGuard {
    const int* status = nullptr;
    const unsigned long long* actual = nullptr;
    const unsigned long long* expected = nullptr;
    __device__ bool valid(int tile) const {
        return !status || (status[tile] == 0 && actual[tile] == expected[tile]);
    }
};

DecodeGuard decode_guard(PyObject* object) {
    if (object == Py_None) {
        return {};
    }
    if (!PyTuple_Check(object) || PyTuple_Size(object) != 3) {
        throw std::invalid_argument("invalid FITS decode guard");
    }
    std::array<void*, 3> pointers;
    for (int i = 0; i < 3; ++i) {
        pointers[i] = PyLong_AsVoidPtr(PyTuple_GetItem(object, i));
        if (PyErr_Occurred() || !pointers[i]) {
            PyErr_Clear();
            throw std::invalid_argument("invalid FITS decode guard pointer");
        }
        cudaPointerAttributes attributes{};
        check(cudaPointerGetAttributes(&attributes, pointers[i]));
        int device;
        check(cudaGetDevice(&device));
        if (attributes.type != cudaMemoryTypeDevice
            || attributes.device != device) {
            throw std::invalid_argument(
                "FITS decode guard must reside on the active CUDA device");
        }
    }
    return {static_cast<const int*>(pointers[0]),
        static_cast<const unsigned long long*>(pointers[1]),
        static_cast<const unsigned long long*>(pointers[2])};
}

struct TileGeometry {
    const int *row, *column, *height, *width, *source_row, *source_column,
        *source_width;
};

__global__ void restore_kernel(
    const unsigned char* source,
    unsigned char* output,
    const std::int64_t* offsets,
    const std::int64_t* lengths,
    TileGeometry geometry,
    long long stride,
    int itemsize,
    bool shuffled,
    DecodeGuard guard) {
    const int tile = blockIdx.x;
    if (!guard.valid(tile)) {
        return;
    }
    const int height = geometry.height[tile], width = geometry.width[tile];
    const long long count = lengths[tile] / itemsize;
    for (int i = threadIdx.x; i < height * width; i += blockDim.x) {
        const int row = i / width, column = i % width;
        const long long pixel =
            (geometry.source_row[tile] + row) * geometry.source_width[tile]
            + geometry.source_column[tile] + column;
        auto* destination = output + (geometry.row[tile] + row) * stride
            + (geometry.column[tile] + column) * itemsize;
        for (int byte = 0; byte < itemsize; ++byte) {
            const int source_byte = itemsize - 1 - byte;
            const long long position = shuffled
                ? source_byte * count + pixel
                : pixel * itemsize + source_byte;
            destination[byte] = source[offsets[tile] + position];
        }
    }
}

template <class Integer, class Real>
__global__ void dequantize_kernel(
    const Integer* input,
    Real* output,
    TileGeometry geometry,
    const double* scales,
    const double* zeros,
    long long stride,
    DecodeGuard guard) {
    const int tile = blockIdx.x;
    if (!guard.valid(tile)) {
        return;
    }
    const int width = geometry.width[tile];
    const Real scale = static_cast<Real>(scales[tile]),
               zero = static_cast<Real>(zeros[tile]);
    for (int i = threadIdx.x; i < geometry.height[tile] * width;
        i += blockDim.x) {
        const long long index = (geometry.row[tile] + i / width) * stride
            + geometry.column[tile] + i % width;
        // The existing RawKernel expression contracts this multiply and add.
        if constexpr (sizeof(Real) == 4) {
            output[index] =
                __fmaf_rn(static_cast<Real>(input[index]), scale, zero);
        } else {
            output[index] =
                __fma_rn(static_cast<Real>(input[index]), scale, zero);
        }
    }
}

std::vector<Array> array_arguments(PyObject* values, int count) {
    if (!PyTuple_Check(values) || PyTuple_Size(values) != count) {
        throw std::invalid_argument("invalid FITS tile argument tuple");
    }
    std::vector<Array> arrays;
    arrays.reserve(count);
    for (int i = 0; i < count; ++i) {
        arrays.emplace_back(PyTuple_GetItem(values, i));
    }
    return arrays;
}
}  // namespace

PyObject* restore_fits_tiles(PyObject*, PyObject* arguments) {
    PyObject *values, *guard_object;
    int itemsize, shuffled;
    unsigned long long stream_value;
    if (!PyArg_ParseTuple(
            arguments,
            "OipOK:restore_fits_tiles",
            &values,
            &itemsize,
            &shuffled,
            &guard_object,
            &stream_value)) {
        return nullptr;
    }
    try {
        auto arrays = array_arguments(values, 11);
        const auto guard = decode_guard(guard_object);
        arrays[0].require('u', 1, 1);
        if (!arrays[1].writable || arrays[1].shape.size() != 2
            || arrays[1].strides[1] != itemsize
            || arrays[1].itemsize != itemsize
            || (itemsize != 1 && itemsize != 2 && itemsize != 4
                && itemsize != 8)) {
            throw std::invalid_argument("invalid native FITS output layout");
        }
        arrays[2].require('i', 8, 1);
        arrays[3].require('i', 8, 1);
        const auto tiles = arrays[2].size;
        for (int i = 3; i < 11; ++i) {
            if (i >= 4) {
                arrays[i].require('i', 4, 1);
            }
            if (arrays[i].size != tiles) {
                throw std::invalid_argument(
                    "FITS tile tables must have equal length");
            }
        }
        TileGeometry geometry{arrays[4].data<int>(),
            arrays[5].data<int>(),
            arrays[6].data<int>(),
            arrays[7].data<int>(),
            arrays[8].data<int>(),
            arrays[9].data<int>(),
            arrays[10].data<int>()};
        Detached detached;
        StreamGuard stream(stream_value);
        for (const auto& array : arrays) {
            array.wait(stream.stream);
        }
        if (tiles) {
            restore_kernel<<<tiles, 256, 0, stream.stream>>>(
                arrays[0].data<unsigned char>(),
                arrays[1].data<unsigned char>(),
                arrays[2].data<std::int64_t>(),
                arrays[3].data<std::int64_t>(),
                geometry,
                arrays[1].strides[0],
                itemsize,
                shuffled,
                guard);
        }
        stream.finish();
    } catch (...) {
        return translate_exception();
    }
    Py_RETURN_NONE;
}

PyObject* dequantize_fits_tiles(PyObject*, PyObject* arguments) {
    PyObject *values, *guard_object;
    unsigned long long stream_value;
    if (!PyArg_ParseTuple(
            arguments,
            "OOK:dequantize_fits_tiles",
            &values,
            &guard_object,
            &stream_value)) {
        return nullptr;
    }
    try {
        auto arrays = array_arguments(values, 8);
        const auto guard = decode_guard(guard_object);
        if (arrays[0].kind != 'i' || arrays[1].kind != 'f'
            || (arrays[0].itemsize != 4 && arrays[0].itemsize != 8)
            || (arrays[1].itemsize != 4 && arrays[1].itemsize != 8)
            || arrays[0].shape.size() != 2 || arrays[0].shape != arrays[1].shape
            || !arrays[0].contiguous() || !arrays[1].contiguous()
            || !arrays[1].writable) {
            throw std::invalid_argument(
                "invalid native FITS dequantization buffers");
        }
        const auto tiles = arrays[2].size;
        for (int i = 2; i < 8; ++i) {
            arrays[i].require(i < 6 ? 'i' : 'f', i < 6 ? 4 : 8, 1);
            if (arrays[i].size != tiles) {
                throw std::invalid_argument(
                    "FITS tile tables must have equal length");
            }
        }
        TileGeometry geometry{arrays[2].data<int>(),
            arrays[3].data<int>(),
            arrays[4].data<int>(),
            arrays[5].data<int>(),
            nullptr,
            nullptr,
            nullptr};
        Detached detached;
        StreamGuard stream(stream_value);
        for (const auto& array : arrays) {
            array.wait(stream.stream);
        }
#define DEQUANTIZE(I, R)                                                       \
    dequantize_kernel<I, R><<<tiles, 256, 0, stream.stream>>>(                 \
        arrays[0].data<I>(),                                                   \
        arrays[1].data<R>(),                                                   \
        geometry,                                                              \
        arrays[6].data<double>(),                                              \
        arrays[7].data<double>(),                                              \
        arrays[1].shape[1],                                                    \
        guard)
        if (tiles) {
            if (arrays[0].itemsize == 4 && arrays[1].itemsize == 4) {
                DEQUANTIZE(int, float);
            } else if (arrays[0].itemsize == 4) {
                DEQUANTIZE(int, double);
            } else if (arrays[1].itemsize == 4) {
                DEQUANTIZE(long long, float);
            } else {
                DEQUANTIZE(long long, double);
            }
        }
#undef DEQUANTIZE
        stream.finish();
    } catch (...) {
        return translate_exception();
    }
    Py_RETURN_NONE;
}
}  // namespace cuphoton::imaging
