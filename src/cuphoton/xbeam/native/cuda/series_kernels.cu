// SPDX-License-Identifier: Apache-2.0
#include "xbeam/series.hpp"
#include "gpu.hpp"
#include <cmath>
namespace xbeam {
namespace {
template <class Complex>
__global__ void pack(
    const unsigned char* raw,
    const unsigned char* mask,
    int dtype,
    int rows,
    int samples,
    int fft_n,
    long long hop,
    Complex* out,
    unsigned char* valid,
    double2* masks,
    int* error) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        i < (long long)rows * fft_n;
        i += (long long)gridDim.x * blockDim.x) {
        const int row = i / fft_n, col = i % fft_n;
        double re = 0, im = 0;
        bool live = false;
        if (col < samples) {
            const long long source = row * hop + col;
            const auto flag = mask ? mask[source] : 1;
            if (flag > 1) {
                atomicOr(error, 2);
            }
            live = flag == 1;
            if (live) {
                if (dtype == 0) {
                    re = reinterpret_cast<const float*>(raw)[source];
                } else if (dtype == 1) {
                    re = reinterpret_cast<const double*>(raw)[source];
                } else if (dtype == 2) {
                    const auto value =
                        reinterpret_cast<const float2*>(raw)[source];
                    re = value.x;
                    im = value.y;
                } else {
                    const auto value =
                        reinterpret_cast<const double2*>(raw)[source];
                    re = value.x;
                    im = value.y;
                }
                if (!isfinite(re) || !isfinite(im)) {
                    atomicOr(error, 1);
                }
            }
            valid[(long long)row * samples + col] = live;
        }
        Complex value;
        value.x = re;
        value.y = im;
        if (live
            && (!isfinite(double(value.x)) || !isfinite(double(value.y)))) {
            atomicOr(error, 1);
        }
        out[i] = value;
        if (masks) {
            masks[i] = make_double2(live ? 1 : 0, 0);
        }
    }
}
__global__ void stats(
    const unsigned char* a,
    const unsigned char* b,
    int rows,
    int samples,
    unsigned long long* out) {
    const int row = blockIdx.x;
    unsigned long long overlap = 0, first = 0, second = 0;
    for (int i = threadIdx.x; i < samples; i += blockDim.x) {
        const auto offset = (long long)row * samples + i;
        first += a[offset] != 0;
        second += b[offset] != 0;
        overlap += a[offset] && b[offset];
    }
    __shared__ unsigned long long values[3][256];
    values[0][threadIdx.x] = overlap;
    values[1][threadIdx.x] = first;
    values[2][threadIdx.x] = second;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (threadIdx.x < stride) {
            for (int k = 0; k < 3; ++k) {
                values[k][threadIdx.x] += values[k][threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    if (!threadIdx.x) {
        atomicAdd(out, (unsigned long long)(values[1][0] && values[2][0]));
        atomicAdd(out + 1, values[0][0]);
    }
}
template <class Complex>
__global__ void add(int n, const Complex* a, Complex* b) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
        i += gridDim.x * blockDim.x) {
        b[i].x += a[i].x;
        b[i].y += a[i].y;
    }
}
template <class Complex>
__global__ void gather(
    int pairs,
    int fft_n,
    int first,
    int count,
    const Complex* src,
    Complex* dst,
    const double2* masks,
    long long* coverage) {
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
        i < (long long)pairs * count;
        i += (long long)gridDim.x * blockDim.x) {
        const int lag = first + i % count;
        const int bin = ((-lag) % fft_n + fft_n) % fft_n;
        const auto index = (i / count) * fft_n + bin;
        Complex value;
        value.x = src[index].x / fft_n;
        value.y = src[index].y / fft_n;
        dst[i] = value;
        if (masks) {
            coverage[i] = llround(masks[index].x / fft_n);
        }
    }
}
}  // namespace
void pack_series_input(
    const void* raw,
    const unsigned char* mask,
    int dtype,
    bool wide,
    int rows,
    int samples,
    int fft_n,
    std::int64_t hop,
    void* output,
    unsigned char* validity,
    void* masks,
    int* error,
    std::uintptr_t stream) {
    const auto s = reinterpret_cast<cudaStream_t>(stream);
    if (wide) {
        pack<<<256, 256, 0, s>>>(
            static_cast<const unsigned char*>(raw),
            mask,
            dtype,
            rows,
            samples,
            fft_n,
            hop,
            static_cast<double2*>(output),
            validity,
            static_cast<double2*>(masks),
            error);
    } else {
        pack<<<256, 256, 0, s>>>(
            static_cast<const unsigned char*>(raw),
            mask,
            dtype,
            rows,
            samples,
            fft_n,
            hop,
            static_cast<float2*>(output),
            validity,
            static_cast<double2*>(masks),
            error);
    }
    gpu::check(cudaGetLastError());
}
void series_statistics(
    const unsigned char* a,
    const unsigned char* b,
    int rows,
    int samples,
    std::int64_t* out,
    std::uintptr_t stream) {
    stats<<<rows, 256, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        a, b, rows, samples, reinterpret_cast<unsigned long long*>(out));
    gpu::check(cudaGetLastError());
}
void add_series_spectrum(
    bool wide,
    int channels,
    const void* input,
    void* output,
    std::uintptr_t stream) {
    const auto s = reinterpret_cast<cudaStream_t>(stream);
    if (wide) {
        add<<<256, 256, 0, s>>>(
            channels,
            static_cast<const double2*>(input),
            static_cast<double2*>(output));
    } else {
        add<<<256, 256, 0, s>>>(
            channels,
            static_cast<const float2*>(input),
            static_cast<float2*>(output));
    }
    gpu::check(cudaGetLastError());
}
void gather_series_lag(
    bool wide,
    int pairs,
    int fft_n,
    int first,
    int count,
    const void* inverse,
    void* out,
    const void* masks,
    std::int64_t* coverage,
    std::uintptr_t stream) {
    const auto s = reinterpret_cast<cudaStream_t>(stream);
    if (wide) {
        gather<<<256, 256, 0, s>>>(
            pairs,
            fft_n,
            first,
            count,
            static_cast<const double2*>(inverse),
            static_cast<double2*>(out),
            static_cast<const double2*>(masks),
            reinterpret_cast<long long*>(coverage));
    } else {
        gather<<<256, 256, 0, s>>>(
            pairs,
            fft_n,
            first,
            count,
            static_cast<const float2*>(inverse),
            static_cast<float2*>(out),
            static_cast<const double2*>(masks),
            reinterpret_cast<long long*>(coverage));
    }
    gpu::check(cudaGetLastError());
}
}  // namespace xbeam
