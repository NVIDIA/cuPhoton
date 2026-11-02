/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "imaging_api.h"
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <algorithm>
#include <climits>
#include <cmath>
#include <memory>

namespace cuphoton::imaging {
namespace {

struct Candidate {
    double score, compact_fraction, negative_radial;
    std::int64_t y, x;
};
static_assert(sizeof(Candidate) == 40);

__global__ void validity(
    const double* image,
    const double* variance,
    const bool* mask,
    bool* valid,
    int* count,
    double* sortable,
    int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) {
        return;
    }
    const bool good = isfinite(image[i])
        && (!variance || (isfinite(variance[i]) && variance[i] > 0))
        && (!mask || mask[i]);
    valid[i] = good;
    sortable[i] = good ? image[i] : CUDART_INF;
    if (good) {
        atomicAdd(count, 1);
    }
}

// In-place order statistic; every window has odd cardinality. Duplicate
// values progress in both partition directions, including constant images.
__device__ double median(double* values, int size) {
    const int middle = size / 2;
    int lower = 0, upper = size - 1;
    while (lower < upper) {
        const double pivot = values[(lower + upper) / 2];
        int left = lower, right = upper;
        while (left <= right) {
            while (values[left] < pivot) {
                ++left;
            }
            while (values[right] > pivot) {
                --right;
            }
            if (left <= right) {
                const double value = values[left];
                values[left] = values[right];
                values[right] = value;
                ++left;
                --right;
            }
        }
        if (middle <= right) {
            upper = right;
        } else if (middle >= left) {
            lower = left;
        } else {
            break;
        }
    }
    return values[middle];
}

template <int Capacity>
__global__ void highpass_kernel(
    const double* image,
    const bool* valid,
    const double* sorted,
    int valid_count,
    double* highpass,
    double* sortable,
    int height,
    int width,
    int filter) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= height * width) {
        return;
    }
    if (!valid[i]) {
        highpass[i] = -CUDART_INF;
        sortable[i] = CUDART_INF;
        return;
    }
    const int y = i / width, x = i % width, radius = filter / 2;
    const double fill = valid_count % 2
        ? sorted[valid_count / 2]
        : (sorted[valid_count / 2 - 1] + sorted[valid_count / 2]) / 2.0;
    double window[Capacity];
    int at = 0;
    for (int dy = -radius; dy <= radius; ++dy) {
        const int row = min(max(y + dy, 0), height - 1);
        for (int dx = -radius; dx <= radius; ++dx) {
            const int column = min(max(x + dx, 0), width - 1);
            const int source = row * width + column;
            window[at++] = valid[source] ? image[source] : fill;
        }
    }
    highpass[i] = image[i] - median(window, filter * filter);
    sortable[i] = highpass[i];
}

__global__ void threshold_kernel(
    const double* sorted, int count, double percentile, double* output) {
    const double index = (percentile / 100.0) * (count - 1.0);
    const int lower = static_cast<int>(floor(index));
    const int upper = min(lower + 1, count - 1);
    const double fraction = index - lower;
    const double difference = sorted[upper] - sorted[lower];
    // CuPy's percentile elementwise expression contracts this multiply/add.
    *output = fraction < 0.5 ? fma(difference, fraction, sorted[lower])
                             : fma(-difference, 1.0 - fraction, sorted[upper]);
}

__global__ void peak_kernel(
    const double* highpass,
    const bool* valid,
    const double* threshold,
    bool* peaks,
    int height,
    int width,
    int filter) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= height * width) {
        return;
    }
    const int y = i / width, x = i % width;
    bool keep = valid[i] && highpass[i] > *threshold;
    const int radius = filter / 2;
    if (keep) {
        for (int dy = -radius; dy <= radius && keep; ++dy) {
            const int row = min(max(y + dy, 0), height - 1);
            for (int dx = -radius; dx <= radius; ++dx) {
                const int column = min(max(x + dx, 0), width - 1);
                if (highpass[row * width + column] > highpass[i]) {
                    keep = false;
                    break;
                }
            }
        }
    }
    peaks[i] = keep;
}

template <int Capacity>
__global__ void score_kernel(
    const double* image,
    const double* variance,
    const bool* valid,
    const int* peaks,
    Candidate* candidates,
    bool* keep,
    int count,
    int height,
    int width,
    int stamp,
    int core) {
    const int candidate = blockIdx.x;
    if (candidate >= count) {
        return;
    }
    const int index = peaks[candidate], y = index / width, x = index % width;
    const int half = stamp / 2;
    if (y < half || x < half || y + half >= height || x + half >= width) {
        if (!threadIdx.x) {
            keep[candidate] = false;
        }
        return;
    }
    __shared__ double window[Capacity];
    __shared__ double background;
    __shared__ int invalid;
    if (!threadIdx.x) {
        invalid = 0;
    }
    __syncthreads();
    for (int pixel = threadIdx.x; pixel < stamp * stamp; pixel += blockDim.x) {
        const int source =
            (y + pixel / stamp - half) * width + x + pixel % stamp - half;
        window[pixel] = image[source];
        if (!valid[source]) {
            atomicExch(&invalid, 1);
        }
    }
    __syncthreads();
    if (invalid) {
        if (!threadIdx.x) {
            keep[candidate] = false;
        }
        return;
    }
    if (!threadIdx.x) {
        background = median(window, stamp * stamp);
    }
    __syncthreads();
    double total = 0, center = 0, radial = 0;
    for (int pixel = threadIdx.x; pixel < stamp * stamp; pixel += blockDim.x) {
        const int dy = pixel / stamp - half, dx = pixel % stamp - half;
        const double weight =
            fmax(image[(y + dy) * width + x + dx] - background, 0.0);
        total += weight;
        if (abs(dy) <= core / 2 && abs(dx) <= core / 2) {
            center += weight;
        }
        radial += weight * (dx * dx + dy * dy);
    }
    using Reduce = cub::BlockReduce<double, 256>;
    __shared__ typename Reduce::TempStorage storage;
    total = Reduce(storage).Sum(total);
    __syncthreads();
    center = Reduce(storage).Sum(center);
    __syncthreads();
    radial = Reduce(storage).Sum(radial);
    if (!threadIdx.x) {
        keep[candidate] = total > 0.0;
        const double signal = image[index] - background;
        candidates[candidate] = {
            variance ? signal / sqrt(variance[index]) : signal,
            center / total,
            -(radial / total),
            y,
            x};
    }
}

__global__ void stamp_bits(
    const std::int64_t* boxes, int count, unsigned int* bits, int width) {
    const int box = blockIdx.x;
    if (box >= count) {
        return;
    }
    const auto* bounds = boxes + 4 * box;
    const int height = bounds[1] - bounds[0], span = bounds[3] - bounds[2];
    for (int pixel = threadIdx.x; pixel < height * span; pixel += blockDim.x) {
        const int index =
            (bounds[0] + pixel / span) * width + bounds[2] + pixel % span;
        atomicOr(bits + index / 32, 1U << (index % 32));
    }
}

__global__ void expand_bits(const unsigned int* bits, bool* output, int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = (bits[i / 32] >> (i % 32)) & 1U;
    }
}

void sort_values(
    double* input, double* output, int count, cudaStream_t stream) {
    std::size_t bytes = 0;
    check(
        cub::DeviceRadixSort::SortKeys(
            nullptr, bytes, input, output, count, 0, 64, stream));
    DeviceBuffer<char> temporary(bytes);
    check(
        cub::DeviceRadixSort::SortKeys(
            temporary.get(), bytes, input, output, count, 0, 64, stream));
}
}  // namespace

PyObject* selection_candidates(PyObject*, PyObject* arguments) {
    PyObject *image_object, *variance_object, *mask_object;
    int background_filter, peak_filter, stamp, core;
    double percentile;
    unsigned long long stream_value;
    if (!PyArg_ParseTuple(
            arguments,
            "OOOiiiidK:selection_candidates",
            &image_object,
            &variance_object,
            &mask_object,
            &background_filter,
            &peak_filter,
            &stamp,
            &core,
            &percentile,
            &stream_value)) {
        return nullptr;
    }
    try {
        Array image(image_object);
        image.require('f', 8, 2);
        if (image.size > INT_MAX || image.size == 0 || background_filter < 1
            || background_filter > 31 || !(background_filter % 2)
            || peak_filter < 1 || !(peak_filter % 2) || stamp < 1 || stamp > 63
            || !(stamp % 2) || core < 1 || core > stamp || !(core % 2)
            || !(percentile > 0 && percentile < 100)) {
            throw std::invalid_argument(
                "unsupported native compact-source selection dimensions");
        }
        std::unique_ptr<Array> variance, mask;
        if (variance_object != Py_None) {
            variance = std::make_unique<Array>(variance_object);
            variance->require('f', 8, 2);
            if (variance->shape != image.shape) {
                throw std::invalid_argument(
                    "variance must match the image shape");
            }
        }
        if (mask_object != Py_None) {
            mask = std::make_unique<Array>(mask_object);
            mask->require('b', 1, 2);
            if (mask->shape != image.shape) {
                throw std::invalid_argument(
                    "valid_mask must match the image shape");
            }
        }
        std::vector<Candidate> host_candidates;
        double host_threshold;
        {
            Detached detached;
            StreamGuard stream(stream_value);
            image.wait(stream.stream);
            if (variance) {
                variance->wait(stream.stream);
            }
            if (mask) {
                mask->wait(stream.stream);
            }
            const int size = image.size, height = image.shape[0],
                      width = image.shape[1];
            const int blocks = (size + 255) / 256;
            DeviceBuffer<bool> valid(size), peaks(size);
            DeviceBuffer<double> sortable(size), sorted(size), highpass(size),
                threshold(1);
            DeviceBuffer<int> count(1), indices(size);
            check(cudaMemsetAsync(count.get(), 0, sizeof(int), stream.stream));
            validity<<<blocks, 256, 0, stream.stream>>>(
                image.data<double>(),
                variance ? variance->data<double>() : nullptr,
                mask ? mask->data<bool>() : nullptr,
                valid.get(),
                count.get(),
                sortable.get(),
                size);
            int valid_count;
            check(cudaMemcpyAsync(
                &valid_count,
                count.get(),
                sizeof(int),
                cudaMemcpyDeviceToHost,
                stream.stream));
            stream.finish();
            if (!valid_count) {
                throw std::invalid_argument(
                    "no finite pixels remain for compact-source selection");
            }
            sort_values(sortable.get(), sorted.get(), size, stream.stream);
            // Smaller blocks bound the local storage used by median windows.
            const int highpass_blocks = (size + 127) / 128;
#define HIGHPASS(CAPACITY)                                                     \
    highpass_kernel<CAPACITY><<<highpass_blocks, 128, 0, stream.stream>>>(     \
        image.data<double>(),                                                  \
        valid.get(),                                                           \
        sorted.get(),                                                          \
        valid_count,                                                           \
        highpass.get(),                                                        \
        sortable.get(),                                                        \
        height,                                                                \
        width,                                                                 \
        background_filter)
            if (background_filter <= 3) {
                HIGHPASS(9);
            } else if (background_filter <= 7) {
                HIGHPASS(49);
            } else if (background_filter <= 15) {
                HIGHPASS(225);
            } else {
                HIGHPASS(961);
            }
#undef HIGHPASS
            sort_values(sortable.get(), sorted.get(), size, stream.stream);
            threshold_kernel<<<1, 1, 0, stream.stream>>>(
                sorted.get(), valid_count, percentile, threshold.get());
            peak_kernel<<<blocks, 256, 0, stream.stream>>>(
                highpass.get(),
                valid.get(),
                threshold.get(),
                peaks.get(),
                height,
                width,
                peak_filter);
            std::size_t bytes = 0;
            auto counter = thrust::make_counting_iterator<int>(0);
            check(
                cub::DeviceSelect::Flagged(
                    nullptr,
                    bytes,
                    counter,
                    peaks.get(),
                    indices.get(),
                    count.get(),
                    size,
                    stream.stream));
            DeviceBuffer<char> select_storage(bytes);
            check(
                cub::DeviceSelect::Flagged(
                    select_storage.get(),
                    bytes,
                    counter,
                    peaks.get(),
                    indices.get(),
                    count.get(),
                    size,
                    stream.stream));
            int peak_count;
            check(cudaMemcpyAsync(
                &peak_count,
                count.get(),
                sizeof(int),
                cudaMemcpyDeviceToHost,
                stream.stream));
            stream.finish();
            if (!peak_count) {
                throw std::invalid_argument(
                    "no compact-source peaks found for the requested mask");
            }
            DeviceBuffer<Candidate> candidates(peak_count),
                retained(peak_count);
            DeviceBuffer<bool> keep(peak_count);
#define SCORE(CAPACITY)                                                        \
    score_kernel<CAPACITY><<<peak_count, 256, 0, stream.stream>>>(             \
        image.data<double>(),                                                  \
        variance ? variance->data<double>() : nullptr,                         \
        valid.get(),                                                           \
        indices.get(),                                                         \
        candidates.get(),                                                      \
        keep.get(),                                                            \
        peak_count,                                                            \
        height,                                                                \
        width,                                                                 \
        stamp,                                                                 \
        core)
            if (stamp <= 15) {
                SCORE(225);
            } else if (stamp <= 31) {
                SCORE(961);
            } else {
                SCORE(3969);
            }
#undef SCORE
            bytes = 0;
            check(
                cub::DeviceSelect::Flagged(
                    nullptr,
                    bytes,
                    candidates.get(),
                    keep.get(),
                    retained.get(),
                    count.get(),
                    peak_count,
                    stream.stream));
            DeviceBuffer<char> final_storage(bytes);
            check(
                cub::DeviceSelect::Flagged(
                    final_storage.get(),
                    bytes,
                    candidates.get(),
                    keep.get(),
                    retained.get(),
                    count.get(),
                    peak_count,
                    stream.stream));
            int candidate_count;
            check(cudaMemcpyAsync(
                &candidate_count,
                count.get(),
                sizeof(int),
                cudaMemcpyDeviceToHost,
                stream.stream));
            check(cudaMemcpyAsync(
                &host_threshold,
                threshold.get(),
                sizeof(double),
                cudaMemcpyDeviceToHost,
                stream.stream));
            stream.finish();
            if (!candidate_count) {
                throw std::invalid_argument(
                    "no valid compact-source stamps remain after finite-mask filtering");
            }
            host_candidates.resize(candidate_count);
            check(cudaMemcpyAsync(
                host_candidates.data(),
                retained.get(),
                candidate_count * sizeof(Candidate),
                cudaMemcpyDeviceToHost,
                stream.stream));
            stream.finish();
        }
        PyObject* records = PyBytes_FromStringAndSize(
            reinterpret_cast<const char*>(host_candidates.data()),
            host_candidates.size() * sizeof(Candidate));
        if (!records) {
            return nullptr;
        }
        return Py_BuildValue("Nd", records, host_threshold);
    } catch (...) {
        return translate_exception();
    }
}

PyObject* stamp_mask(PyObject*, PyObject* arguments) {
    PyObject *boxes_object, *mask_object;
    unsigned long long stream_value;
    if (!PyArg_ParseTuple(
            arguments,
            "OOK:stamp_mask",
            &boxes_object,
            &mask_object,
            &stream_value)) {
        return nullptr;
    }
    try {
        Array boxes(boxes_object), mask(mask_object);
        boxes.require('i', 8, 2);
        mask.require('b', 1, 2, true);
        if (boxes.shape[1] != 4 || mask.size > INT_MAX) {
            throw std::invalid_argument("invalid stamp-mask dimensions");
        }
        std::vector<std::int64_t> host_boxes(boxes.size);
        Detached detached;
        StreamGuard stream(stream_value);
        boxes.wait(stream.stream);
        mask.wait(stream.stream);
        check(cudaMemcpyAsync(
            host_boxes.data(),
            boxes.pointer,
            boxes.size * sizeof(std::int64_t),
            cudaMemcpyDeviceToHost,
            stream.stream));
        stream.finish();
        for (std::size_t i = 0; i < host_boxes.size(); i += 4) {
            if (host_boxes[i] < 0 || host_boxes[i + 1] > mask.shape[0]
                || host_boxes[i + 2] < 0 || host_boxes[i + 3] > mask.shape[1]
                || host_boxes[i] > host_boxes[i + 1]
                || host_boxes[i + 2] > host_boxes[i + 3]) {
                throw std::invalid_argument(
                    "stamp bounds lie outside the mask");
            }
        }
        DeviceBuffer<unsigned int> bits((mask.size + 31) / 32);
        check(cudaMemsetAsync(
            bits.get(), 0, bits.count * sizeof(unsigned int), stream.stream));
        if (boxes.shape[0]) {
            stamp_bits<<<boxes.shape[0], 256, 0, stream.stream>>>(
                boxes.data<std::int64_t>(),
                boxes.shape[0],
                bits.get(),
                mask.shape[1]);
        }
        if (mask.size) {
            expand_bits<<<(mask.size + 255) / 256, 256, 0, stream.stream>>>(
                bits.get(), mask.data<bool>(), mask.size);
        }
        stream.finish();
    } catch (...) {
        return translate_exception();
    }
    Py_RETURN_NONE;
}
}  // namespace cuphoton::imaging
