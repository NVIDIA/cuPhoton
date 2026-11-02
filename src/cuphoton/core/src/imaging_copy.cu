/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "imaging_api.h"
#include <cuda_fp16.h>
#include <memory>

namespace cuphoton::imaging {
namespace {
struct Layout {
    long long shape[8];
    long long strides[8];
    int ndim;
};

template <class T>
__device__ T read_value(const char* pointer, char kind, int bytes) {
    if (kind == 'f') {
        if (bytes == 2) {
            return static_cast<T>(
                __half2float(*reinterpret_cast<const __half*>(pointer)));
        }
        return bytes == 4
            ? static_cast<T>(*reinterpret_cast<const float*>(pointer))
            : static_cast<T>(*reinterpret_cast<const double*>(pointer));
    }
    if (kind == 'i') {
        switch (bytes) {
        case 1:
            return static_cast<T>(
                *reinterpret_cast<const signed char*>(pointer));
        case 2:
            return static_cast<T>(*reinterpret_cast<const short*>(pointer));
        case 4:
            return static_cast<T>(*reinterpret_cast<const int*>(pointer));
        default:
            return static_cast<T>(*reinterpret_cast<const long long*>(pointer));
        }
    }
    switch (bytes) {
    case 1:
        return static_cast<T>(*reinterpret_cast<const unsigned char*>(pointer));
    case 2:
        return static_cast<T>(
            *reinterpret_cast<const unsigned short*>(pointer));
    case 4:
        return static_cast<T>(*reinterpret_cast<const unsigned int*>(pointer));
    default:
        return static_cast<T>(
            *reinterpret_cast<const unsigned long long*>(pointer));
    }
}

template <class T>
__global__ void copy_kernel(
    const char* source,
    T* output,
    long long count,
    Layout layout,
    char kind,
    int bytes) {
    const long long index =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count) {
        return;
    }
    long long remaining = index;
    long long offset = 0;
    for (int axis = layout.ndim - 1; axis >= 0; --axis) {
        offset += (remaining % layout.shape[axis]) * layout.strides[axis];
        remaining /= layout.shape[axis];
    }
    output[index] = read_value<T>(source + offset, kind, bytes);
}

template <class T>
void launch_copy(
    const Array& input,
    const Array& output,
    Layout layout,
    cudaStream_t stream) {
    copy_kernel<<<(input.size + 255) / 256, 256, 0, stream>>>(
        input.data<char>(),
        output.data<T>(),
        input.size,
        layout,
        input.kind,
        input.itemsize);
}

__global__ void mask_validity(
    const char* input, long long size, char kind, int bytes, int* invalid) {
    const long long index =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= size) {
        return;
    }
    const double value = read_value<double>(input + index * bytes, kind, bytes);
    if (!isfinite(value)) {
        atomicOr(invalid, 1);
    } else if (value != 0.0 && value != 1.0) {
        atomicOr(invalid, 2);
    }
}

__global__ void gather_byte_kernel(
    const unsigned char* input,
    const long long* indices,
    unsigned char* output,
    long long input_size,
    long long count,
    int* invalid) {
    const long long index =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        const long long source = indices[index];
        if (source < 0 || source >= input_size) {
            atomicOr(invalid, 1);
        } else {
            output[index] = input[source];
        }
    }
}

__global__ void selection_mean(
    const double* reference,
    const double* target,
    const bool* mask,
    double* output,
    long long size) {
    const long long index =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < size) {
        output[index] = isfinite(reference[index]) && isfinite(target[index])
                && (!mask || mask[index])
            ? 0.5 * (reference[index] + target[index])
            : CUDART_NAN;
    }
}
}  // namespace

PyObject* gather_bytes(PyObject*, PyObject* arguments) {
    PyObject *input_object, *indices_object, *output_object;
    unsigned long long stream_value;
    if (!PyArg_ParseTuple(
            arguments,
            "OOOK:gather_bytes",
            &input_object,
            &indices_object,
            &output_object,
            &stream_value)) {
        return nullptr;
    }
    try {
        Array input(input_object), indices(indices_object),
            output(output_object);
        input.require('u', 1, 1);
        indices.require('i', 8, 1);
        output.require('u', 1, 1, true);
        if (indices.size != output.size) {
            throw std::invalid_argument("gather output must match index count");
        }
        int invalid = 0;
        if (indices.size) {
            Detached detached;
            StreamGuard stream(stream_value);
            input.wait(stream.stream);
            indices.wait(stream.stream);
            output.wait(stream.stream);
            DeviceBuffer<int> status(1);
            check(cudaMemsetAsync(status.get(), 0, sizeof(int), stream.stream));
            gather_byte_kernel<<<
                (indices.size + 255) / 256,
                256,
                0,
                stream.stream>>>(
                input.data<unsigned char>(),
                indices.data<long long>(),
                output.data<unsigned char>(),
                input.size,
                indices.size,
                status.get());
            check(cudaGetLastError());
            check(cudaMemcpyAsync(
                &invalid,
                status.get(),
                sizeof(int),
                cudaMemcpyDeviceToHost,
                stream.stream));
            stream.finish();
        }
        if (invalid) {
            throw std::invalid_argument("gather index exceeds input buffer");
        }
        Py_RETURN_NONE;
    } catch (...) {
        return translate_exception();
    }
}

PyObject* validate_mask(PyObject*, PyObject* arguments) {
    PyObject* input_object;
    unsigned long long stream_value;
    if (!PyArg_ParseTuple(
            arguments, "OK:validate_mask", &input_object, &stream_value)) {
        return nullptr;
    }
    try {
        Array input(input_object);
        if (!input.contiguous()
            || (input.kind == 'f' && input.itemsize != 2 && input.itemsize != 4
                && input.itemsize != 8)) {
            throw std::invalid_argument("invalid native numeric mask");
        }
        int invalid = 0;
        {
            Detached detached;
            StreamGuard stream(stream_value);
            input.wait(stream.stream);
            DeviceBuffer<int> status(1);
            check(cudaMemsetAsync(status.get(), 0, sizeof(int), stream.stream));
            if (input.size) {
                mask_validity<<<
                    (input.size + 255) / 256,
                    256,
                    0,
                    stream.stream>>>(
                    input.data<char>(),
                    input.size,
                    input.kind,
                    input.itemsize,
                    status.get());
            }
            check(cudaMemcpyAsync(
                &invalid,
                status.get(),
                sizeof(int),
                cudaMemcpyDeviceToHost,
                stream.stream));
            stream.finish();
        }
        return PyLong_FromLong(invalid);
    } catch (...) {
        return translate_exception();
    }
}

PyObject* selection_image(PyObject*, PyObject* arguments) {
    PyObject *reference_object, *target_object, *mask_object, *output_object;
    unsigned long long stream_value;
    if (!PyArg_ParseTuple(
            arguments,
            "OOOOK:selection_image",
            &reference_object,
            &target_object,
            &mask_object,
            &output_object,
            &stream_value)) {
        return nullptr;
    }
    try {
        Array reference(reference_object), target(target_object),
            output(output_object);
        reference.require('f', 8, 2);
        target.require('f', 8, 2);
        output.require('f', 8, 2, true);
        if (reference.shape != target.shape
            || reference.shape != output.shape) {
            throw std::invalid_argument("selection image shapes must match");
        }
        std::unique_ptr<Array> mask;
        if (mask_object != Py_None) {
            mask = std::make_unique<Array>(mask_object);
            mask->require('b', 1, 2);
            if (mask->shape != output.shape) {
                throw std::invalid_argument("selection mask shape must match");
            }
        }
        Detached detached;
        StreamGuard stream(stream_value);
        reference.wait(stream.stream);
        target.wait(stream.stream);
        output.wait(stream.stream);
        if (mask) {
            mask->wait(stream.stream);
        }
        if (output.size) {
            selection_mean<<<
                (output.size + 255) / 256,
                256,
                0,
                stream.stream>>>(
                reference.data<double>(),
                target.data<double>(),
                mask ? mask->data<bool>() : nullptr,
                output.data<double>(),
                output.size);
        }
        stream.finish();
    } catch (...) {
        return translate_exception();
    }
    Py_RETURN_NONE;
}

PyObject* copy_array(PyObject*, PyObject* arguments) {
    PyObject *source, *destination;
    unsigned long long stream_value;
    if (!PyArg_ParseTuple(
            arguments,
            "OOK:copy_array",
            &source,
            &destination,
            &stream_value)) {
        return nullptr;
    }
    try {
        Array input(source), output(destination);
        if (input.shape != output.shape || !output.contiguous()
            || !output.writable
            || (input.kind == 'f' && input.itemsize != 2 && input.itemsize != 4
                && input.itemsize != 8)
            || (output.kind == 'f' && output.itemsize != 2
                && output.itemsize != 4 && output.itemsize != 8)) {
            throw std::invalid_argument("incompatible native array conversion");
        }
        Detached detached;
        StreamGuard stream(stream_value);
        input.wait(stream.stream);
        output.wait(stream.stream);
        if (input.size) {
            if (input.contiguous() && input.kind == output.kind
                && input.itemsize == output.itemsize) {
                check(cudaMemcpyAsync(
                    output.pointer,
                    input.pointer,
                    input.size * input.itemsize,
                    cudaMemcpyDeviceToDevice,
                    stream.stream));
            } else {
                Layout layout{};
                layout.ndim = input.shape.size();
                for (int axis = 0; axis < layout.ndim; ++axis) {
                    layout.shape[axis] = input.shape[axis];
                    layout.strides[axis] = input.strides[axis];
                }
                if (output.kind == 'b') {
                    launch_copy<bool>(input, output, layout, stream.stream);
                } else if (output.kind == 'f' && output.itemsize == 2) {
                    launch_copy<__half>(input, output, layout, stream.stream);
                } else if (output.kind == 'f' && output.itemsize == 4) {
                    launch_copy<float>(input, output, layout, stream.stream);
                } else if (output.kind == 'f') {
                    launch_copy<double>(input, output, layout, stream.stream);
                } else if (output.kind == 'i') {
                    switch (output.itemsize) {
                    case 1:
                        launch_copy<signed char>(
                            input, output, layout, stream.stream);
                        break;
                    case 2:
                        launch_copy<short>(
                            input, output, layout, stream.stream);
                        break;
                    case 4:
                        launch_copy<int>(input, output, layout, stream.stream);
                        break;
                    default:
                        launch_copy<long long>(
                            input, output, layout, stream.stream);
                    }
                } else {
                    switch (output.itemsize) {
                    case 1:
                        launch_copy<unsigned char>(
                            input, output, layout, stream.stream);
                        break;
                    case 2:
                        launch_copy<unsigned short>(
                            input, output, layout, stream.stream);
                        break;
                    case 4:
                        launch_copy<unsigned int>(
                            input, output, layout, stream.stream);
                        break;
                    default:
                        launch_copy<unsigned long long>(
                            input, output, layout, stream.stream);
                    }
                }
            }
        }
        stream.finish();
    } catch (...) {
        return translate_exception();
    }
    Py_RETURN_NONE;
}
}  // namespace cuphoton::imaging
