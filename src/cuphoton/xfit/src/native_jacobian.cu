// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0
//
// Fused native xFit Gaussian dipole Jacobian evaluation and equation packing.
//
// Single-pass fused Jacobian evaluation + packing. The native baseline evaluates
// every selected row into a full-row scratch buffer and then gathers J, J^T and
// the existing residuals in a second launch. Here one launch evaluates each
// observation once and writes all required layouts directly:
//   * packed_jacobian[k, j, m]  -- coalesced per-derivative stores
//   * transposed[k, m, j]       -- two 256-bit stores per observation
//   * compact_residual[k, m]    -- exact copy of residuals[indices[k], m]
//   * finite_row[k]             -- AND over all weighted derivatives of row k
//
// Work decomposition. A thread-block cluster of C CTAs owns R <= 32 selected
// rows (dealt round-robin); CTA rank `part` of the cluster owns a contiguous
// range of 32-pixel chunks of every one of those rows. Pixel warps loop over
// (row, chunk) items, prefetching the next item's weights and residuals (the
// latter copied to compact_residual). Two modes (see the launch geometry):
//   * throughput mode (K >= #SMs): C = 1, whole rows per CTA, the pixel warps
//     first compute the per-row scalars themselves; J^T is staged per warp in
//     shared memory and written as contiguous 256-bit streams.
//   * latency mode (K < #SMs): three helper warps compute the per-row scalars
//     (one lane per row/axis):
//       H0: s = exp(log width) (NaN unless finite and > 0), 1/(s*s)
//       H1: cos, sin of the angle
//       H2: s^cube (runtime pow), RN(1/s^cube)
//     H0/H1 release named barrier A, H2 releases named barrier B; pixel warps
//     wait on A before the lobe evaluation and on B only before the width
//     quotients, so the long pow chain overlaps the per-pixel exp. The helpers
//     also prefetch the parameter array into L2, and J^T records are stored
//     directly with 256-bit stores. For N <= 16 the helpers prepare all source
//     rows directly, removing the index-to-parameter load dependency; pixel
//     warps still use runtime indices to select the requested scalar records.
//   * large short three-plane rows use the lean path: asynchronous shared-memory
//     input buffers and sequential lobes lower register pressure.
// For one-plane throughput, four-warp CTAs serve large short-row cohorts;
// ten-warp CTAs at 96 registers serve long rows, increasing resident warps.
//
// Arithmetic follows Gaussian<T>::observation operation by operation. Every
// multiply/add uses explicit round-to-nearest intrinsics so the compiler cannot
// contract them into FMAs; per-row scalars are hoisted, which does not change
// their rounded values. Divisions by s^3 use a hoisted correctly rounded
// reciprocal plus FMA corrections that reproduce IEEE division bit-for-bit
// (see divide_scaled), with IEEE division as the fallback.

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <algorithm>
#include <cstdint>

#include "native_jacobian.h"

namespace cuphoton::xfit::detail {

cudaError_t launch_jacobian_pack_lean_fp64(
    const double*,
    const double*,
    const int*,
    const double*,
    double*,
    double*,
    double*,
    bool*,
    long long,
    int,
    int,
    int,
    int,
    int,
    cudaStream_t);

namespace cg = cooperative_groups;

namespace {

constexpr int kHelpers = 3;  // helper warps per CTA in latency mode
constexpr int kMaxRows =
    32;  // rows per cluster (finite flags kept as a bit mask)
constexpr int kMaxCluster = 8;
constexpr int kBarA = 1, kBarB = 2;
constexpr long long kPrefetchLines = 4096;  // x prefetch cap (512 KiB)

__device__ __forceinline__ double mul(double a, double b) {
    return __dmul_rn(a, b);
}
__device__ __forceinline__ double add(double a, double b) {
    return __dadd_rn(a, b);
}
__device__ __forceinline__ double sub(double a, double b) {
    return __dsub_rn(a, b);
}

__device__ __forceinline__ void bar_sync(int id, int n) {
    asm volatile("bar.sync %0, %1;" ::"r"(id), "r"(n) : "memory");
}
__device__ __forceinline__ void bar_arrive(int id, int n) {
    asm volatile("bar.arrive %0, %1;" ::"r"(id), "r"(n) : "memory");
}

// 256-bit global store (sm_100+): one full 32-byte sector per thread.
__device__ __forceinline__ void st_v4(
    double* p, double a, double b, double c, double d) {
    asm volatile("st.global.v4.f64 [%0], {%1, %2, %3, %4};" ::"l"(p),
        "d"(a),
        "d"(b),
        "d"(c),
        "d"(d)
        : "memory");
}

// Exponent field (sign masked) of a double, still in place: comparable as an unsigned.
__device__ __forceinline__ unsigned exponent_field(double v) {
    return (unsigned)__double2hiint(v) & 0x7ff00000u;
}
constexpr unsigned kScaledLo = (1023u - 400u) << 20;  // 2^-400
constexpr unsigned kScaledHi = (1023u + 400u)
    << 20;  // 2^400 (inclusive exponent)

__device__ __forceinline__ bool well_scaled(double v) {
    const unsigned e = exponent_field(v);
    return e >= kScaledLo && e <= kScaledHi;
}

// Correctly rounded a / d, bit-identical to IEEE division, for |a| and |d| in
// [2^-400, 2^401). y = RN(1/d) is hoisted per row. One Newton correction makes
// the quotient faithful; the second correction (Markstein's theorem: faithful
// q, y within half an ulp of 1/d, exact FMA remainder) yields RN(a/d). In this
// range no quotient, remainder or product can overflow or lose bits to underflow.
__device__ __forceinline__ double divide_scaled(double a, double d, double y) {
    const double q0 = __dmul_rn(a, y);
    const double q1 = __fma_rn(__fma_rn(-q0, d, a), y, q0);
    return __fma_rn(__fma_rn(-q1, d, a), y, q1);
}

__device__ __forceinline__ double valid_width(double s) {
    return (isfinite(s) && s > 0.0)
        ? s
        : __longlong_as_double(0x7ff8000000000000LL);
}

// Per-row scalars, phase A (needed by the lobe evaluation).
struct RowA {
    double amplitude, cosine, sine, sx, sy, ix, iy, xp, yp, xn, yn;
    long long src;
};
// Per-row scalars, phase B (needed only by the width quotients).
struct RowB {
    double cube[2], rcp[2];
    int fast[2];
};

// exp(a) exactly as the CUDA math library computes it (libdevice __nv_exp,
// transcribed operation by operation from its PTX: round-to-integer via the
// 1.5*2^52 shift, two-part ln2 reduction, degree-11 Horner polynomial, exponent
// add, and the split scaling / overflow / underflow path for |a| >= 708.39).
// Results are bit-identical; the coefficients live in the constant bank so each
// DFMA reads them directly instead of rematerializing 64-bit literals.
__constant__ unsigned long long kExpBits[14] = {0x3FF71547652B82FEull,
    0x4338000000000000ull,
    0xBFE62E42FEFA39EFull,
    0xBC7ABC9E3B39803Full,
    0x3E5ADE1569CE2BDFull,
    0x3E928AF3FCA213EAull,
    0x3EC71DEE62401315ull,
    0x3EFA01997C89EB71ull,
    0x3F2A01A014761F65ull,
    0x3F56C16C1852B7AFull,
    0x3F81111111122322ull,
    0x3FA55555555502A1ull,
    0x3FC5555555555511ull,
    0x3FE000000000000Bull};

__device__ __forceinline__ double exp_c(int i) {
    return __longlong_as_double(kExpBits[i]);
}

__device__ __forceinline__ double exp_native(double a) {
    const double t = __fma_rn(a, exp_c(0), exp_c(1));
    const int i = __double2loint(t);
    const double z = __dsub_rn(t, exp_c(1));
    double f = __fma_rn(z, exp_c(2), a);
    f = __fma_rn(z, exp_c(3), f);
    double p = __fma_rn(f, exp_c(4), exp_c(5));
#pragma unroll
    for (int c = 6; c < 14; ++c) {
        p = __fma_rn(p, f, exp_c(c));
    }
    p = __fma_rn(p, f, 1.0);
    p = __fma_rn(p, f, 1.0);
    const int plo = __double2loint(p), phi = __double2hiint(p);
    double r =
        __hiloint2double((int)((unsigned)phi + ((unsigned)i << 20)), plo);
    const float ah = fabsf(__int_as_float(__double2hiint(a)));
    if (!(ah < __int_as_float(0x4086232B))) {
        r = a < 0.0 ? 0.0
                    : __dadd_rn(a, __longlong_as_double(0x7FF0000000000000ll));
        if (ah < __int_as_float(0x40874800)) {
            const int i1 = (i + (int)((unsigned)i >> 31)) >> 1;
            const double s1 = __hiloint2double(
                (int)((unsigned)phi + ((unsigned)i1 << 20)), plo);
            const double s2 = __hiloint2double(
                (int)(((unsigned)(i - i1) << 20) + 0x3FF00000u), 0);
            r = __dmul_rn(s2, s1);
        }
    }
    return r;
}

// One Gaussian lobe up to (but excluding) the width quotients.
struct Lobe {
    double unit, value, xr, yr, wx_num, wy_num;
};

template <bool kNativeExp>
__device__ __forceinline__ Lobe lobe(
    double amplitude,
    double cosine,
    double sine,
    double ix,
    double iy,
    double cx,
    double cy,
    double ox,
    double oy) {
    const double dx = sub(cx, ox);
    const double dy = sub(cy, oy);
    Lobe l;
    l.xr = add(mul(cosine, dx), mul(sine, dy));
    l.yr = sub(mul(cosine, dy), mul(sine, dx));
    const double arg =
        mul(-0.5, add(mul(mul(l.xr, l.xr), ix), mul(mul(l.yr, l.yr), iy)));
    l.unit = kNativeExp ? exp_native(arg) : exp(arg);
    l.value = mul(amplitude, l.unit);
    l.wx_num = mul(mul(l.value, l.xr), l.xr);
    l.wy_num = mul(mul(l.value, l.yr), l.yr);
    return l;
}

// Remaining lobe derivatives (angle and positions).
__device__ __forceinline__ void lobe_rest(
    const Lobe& l,
    double cosine,
    double sine,
    double ix,
    double iy,
    double ix_minus_iy,
    double& angle,
    double& px,
    double& py) {
    angle = mul(mul(mul(-l.value, l.xr), l.yr), ix_minus_iy);
    px =
        mul(l.value, sub(mul(mul(cosine, l.xr), ix), mul(mul(sine, l.yr), iy)));
    py =
        mul(l.value, add(mul(mul(sine, l.xr), ix), mul(mul(cosine, l.yr), iy)));
}

// Weight one plane, track finiteness, store J (coalesced, one derivative per
// instruction) and J^T. J^T records (64 bytes per pixel) are staged in a per-warp
// shared tile and written back as two contiguous 1 KiB 256-bit-per-lane streams
// (full 128-byte lines) instead of 64-byte-strided half lines. The tile is
// XOR-swizzled at 16-byte granularity (slot c ^ ((record >> 1) & 3)), which
// makes both the record writes and the contiguous read-back bank-conflict free.
template <bool kStage>
__device__ __forceinline__ void emit(
    double (&v)[8],
    double weight,
    double* pj,
    long long M,
    double* tchunk,
    double2* tile,
    int lane,
    int n,
    bool active,
    bool vec_ok,
    unsigned& worst) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        v[j] = mul(v[j], weight);
        worst = max(worst, (unsigned)__double2hiint(v[j]) & 0x7fffffffu);
    }
    if (active) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            pj[j * M] = v[j];
        }
    }
    if (!kStage) {
        // Direct 256-bit stores of this lane's 64-byte record (no staging latency).
        if (active) {
            if (vec_ok) {
                st_v4(tchunk + 8 * lane, v[0], v[1], v[2], v[3]);
                st_v4(tchunk + 8 * lane + 4, v[4], v[5], v[6], v[7]);
            } else {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    tchunk[8 * lane + j] = v[j];
                }
            }
        }
    } else if (vec_ok) {
        const int sw = (lane >> 1) & 3;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            tile[lane * 4 + (c ^ sw)] = make_double2(v[2 * c], v[2 * c + 1]);
        }
        __syncwarp();
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int rec = (lane >> 1)
                + 16 * h;  // pixel record of 32-byte piece lane + 32 h
            if (rec < n) {
                const int sr = (rec >> 1) & 3, c0 = 2 * (lane & 1);
                const double2 a = tile[rec * 4 + (c0 ^ sr)];
                const double2 b = tile[rec * 4 + ((c0 + 1) ^ sr)];
                st_v4(tchunk + 4 * (lane + 32 * h), a.x, a.y, b.x, b.y);
            }
        }
        __syncwarp();
    } else if (active) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            tchunk[8 * lane + j] = v[j];
        }
    }
}

__device__ __forceinline__ void cluster_arrive_relaxed() {
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_arrive() {
    asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
}
__device__ __forceinline__ void cluster_wait() {
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

// Per-row scalar jobs, one lane per row (and width axis):
//   job 0: s = exp(log width) (NaN unless finite and > 0), 1/(s*s), amplitude, centres
//   job 1: cos, sin of the angle
//   job 2: s^cube (runtime pow), RN(1/s^cube), fast-division range flags
__device__ __forceinline__ void row_job(
    int job,
    int lane,
    int R,
    int g,
    int G,
    const double* __restrict__ x,
    const int* __restrict__ indices,
    double cube_exponent,
    RowA* row_a,
    RowB* row_b,
    bool direct) {
    if (job == 0) {
        for (int pair = lane; pair < 2 * R; pair += 32) {
            const int row = pair >> 1, axis = pair & 1;
            const long long src = direct ? row : __ldg(indices + g + row * G);
            const double* xr = x + src * 8;
            const double s = valid_width(exp(__ldg(xr + 1 + axis)));
            const double inv = 1.0 / mul(s, s);
            RowA& o = row_a[row];
            if (axis == 0) {
                o.sx = s;
                o.ix = inv;
                o.amplitude = __ldg(xr + 0);
                o.xp = __ldg(xr + 4);
                o.yp = __ldg(xr + 5);
                o.src = src;
            } else {
                o.sy = s;
                o.iy = inv;
                o.xn = __ldg(xr + 6);
                o.yn = __ldg(xr + 7);
            }
        }
    } else if (job == 1) {
        for (int row = lane; row < R; row += 32) {
            const long long src = direct ? row : __ldg(indices + g + row * G);
            double sn, cs;
            sincos(__ldg(x + src * 8 + 3), &sn, &cs);
            row_a[row].cosine = cs;
            row_a[row].sine = sn;
        }
    } else {
        for (int pair = lane; pair < 2 * R; pair += 32) {
            const int row = pair >> 1, axis = pair & 1;
            const long long src = direct ? row : __ldg(indices + g + row * G);
            const double s = valid_width(exp(__ldg(x + src * 8 + 1 + axis)));
            const double cube = pow(s, cube_exponent);
            const double rcp = 1.0 / cube;
            row_b[row].cube[axis] = cube;
            row_b[row].rcp[axis] = rcp;
            row_b[row].fast[axis] = well_scaled(cube) && well_scaled(rcp);
        }
    }
}

// kThroughput: no helper warps (the pixel warps compute the row scalars) and
// exp with constant-bank coefficients; otherwise three helper warps and the
// immediate-operand library exp (no cold constant-cache miss on the critical path).
template <
    int P,
    int MaxThreads,
    int MinBlocks,
    bool kThroughput,
    bool kDirect = false>
__global__ void __launch_bounds__(MaxThreads, MinBlocks) jacobian_pack_kernel(
    const double* __restrict__ x,
    const double* __restrict__ weights,
    const int* __restrict__ indices,
    const double* __restrict__ residuals,
    double* __restrict__ packed,
    double* __restrict__ transposed,
    double* __restrict__ compact,
    bool* __restrict__ finite_row,
    int K,
    int H,
    int W,
    long long N,
    int G,
    int C,
    unsigned w_magic,
    double cube_exponent,
    int vec_ok_flag) {
    constexpr int helpers = kThroughput ? 0 : kHelpers;
    __shared__ RowA row_a[kMaxRows];
    __shared__ RowB row_b[kMaxRows];
    __shared__ unsigned bad_rows;
    extern __shared__ double2 tiles[];  // [pixel warps][32 records x 4 slots]

    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int NWP = (int)(blockDim.x >> 5) - helpers;  // pixel warps
    const int g = blockIdx.x / C;
    const int part = blockIdx.x - g * C;
    // Rows are dealt round-robin (row slot i of cluster g is packed row g + i * G):
    // clusters advance through their slots at the same pace, so the rows in
    // flight form one dense band of the outputs (better DRAM page locality).
    const int R = (K - g + G - 1) / G;
    const int HW = H * W;
    const long long M = (long long)P * HW;
    const int chunks = (HW + 31) >> 5;
    const int cb = part * chunks / C;
    const int nch = (part + 1) * chunks / C - cb;
    const int pix_hi = min((cb + nch) * 32, HW);
    const bool vec_ok = vec_ok_flag != 0;
    // For tiny source cohorts, all scalar jobs fit in one warp. Loading source
    // rows directly removes the indices -> parameters dependency; pixel warps
    // still select the requested source row using the actual runtime indices.
    constexpr bool preload_all = kDirect;

    if (threadIdx.x == 0) {
        bad_rows = 0u;
    }
    if (C > 1) {
        cluster_arrive();  // publish initialization before remote atomics
    }

    if (warp >= NWP) {
        // ---------------- helper warps (latency mode) ----------------
        const int h = warp - NWP;
        {
            // Data-independent L2 prefetch of the whole parameter array (split over
            // the grid): the dependent x[indices[k]] load then hits L2 instead of
            // paying a second DRAM round trip after the index load.
            const long long lines =
                ((long long)N * 8 * sizeof(double) + 127) / 128;
            if (!preload_all && lines <= kPrefetchLines) {
                const int ht = threadIdx.x - NWP * 32;
                for (long long l = (long long)blockIdx.x * (kHelpers * 32) + ht;
                    l < lines;
                    l += (long long)gridDim.x * (kHelpers * 32)) {
                    asm volatile("prefetch.global.L2 [%0];" ::"l"(x + l * 16));
                }
            }
        }
        row_job(
            h,
            lane,
            preload_all ? N : R,
            g,
            G,
            x,
            indices,
            cube_exponent,
            row_a,
            row_b,
            preload_all);
        if (h < 2) {
            bar_arrive(kBarA, (NWP + 2) * 32);
        } else {
            bar_arrive(kBarB, (NWP + 1) * 32);
        }
    } else {
        // ---------------- pixel warps ----------------
        // Items (row slot r, chunk c) are dealt round-robin to the pixel warps; the
        // counters advance incrementally (no per-item integer division).
        int r = warp / nch, c = warp - (warp / nch) * nch;
        // Pixel row h = pix / W by multiply-high when the host verified it exact.
        auto pixel_row = [&](int pix) {
            return w_magic ? (int)__umulhi((unsigned)pix, w_magic) : pix / W;
        };
        double wn[P], rn[P];
        if (r < R) {
            const int pix = min((cb + c) * 32 + lane, pix_hi - 1);
            const long long src = __ldg(indices + g + r * G);
#pragma unroll
            for (int p = 0; p < P; ++p) {
                wn[p] = __ldg(weights + src * M + (long long)p * HW + pix);
                rn[p] = __ldg(residuals + src * M + (long long)p * HW + pix);
            }
        }
        bool have_b = helpers == 0;
        if (helpers == 0) {
            // Throughput mode: the pixel warps compute the row scalars themselves.
            for (int job = warp; job < 3; job += NWP) {
                row_job(
                    job,
                    lane,
                    preload_all ? N : R,
                    g,
                    G,
                    x,
                    indices,
                    cube_exponent,
                    row_a,
                    row_b,
                    preload_all);
            }
            bar_sync(kBarA, NWP * 32);
        } else {
            bar_sync(kBarA, (NWP + 2) * 32);
        }
        unsigned bad = 0u;
        while (r < R) {
            const int base = (cb + c) * 32;
            const bool active = base + lane < pix_hi;
            const int pix = min(base + lane, pix_hi - 1);
            const int k = g + r * G;
            double wc[P];
#pragma unroll
            for (int p = 0; p < P; ++p) {
                wc[p] = wn[p];
                // Compact residual: exact copy of the selected row.
                if (active) {
                    compact[(long long)k * M + (long long)p * HW + pix] = rn[p];
                }
            }
            // Advance to the next item and prefetch its weights and residuals.
            int nr = r, nc = c + NWP;
            while (nc >= nch) {
                nc -= nch;
                ++nr;
            }
            if (nr < R) {
                const int npix = min((cb + nc) * 32 + lane, pix_hi - 1);
                const long long nsrc =
                    preload_all ? __ldg(indices + g + nr * G) : row_a[nr].src;
#pragma unroll
                for (int p = 0; p < P; ++p) {
                    wn[p] =
                        __ldg(weights + nsrc * M + (long long)p * HW + npix);
                    rn[p] =
                        __ldg(residuals + nsrc * M + (long long)p * HW + npix);
                }
            }
            const int scalar_row = preload_all ? __ldg(indices + k) : r;
            const RowA& ra = row_a[scalar_row];
            const double amplitude = ra.amplitude, cosine = ra.cosine,
                         sine = ra.sine;
            const double ix = ra.ix, iy = ra.iy;
            const int hh = pixel_row(pix);
            const double cx = (double)(pix - hh * W) - 0.5 * (double)(W - 1);
            const double cy = (double)hh - 0.5 * (double)(H - 1);
            const Lobe lp = lobe<kThroughput>(
                amplitude, cosine, sine, ix, iy, cx, cy, ra.xp, ra.yp);
            const Lobe ln = lobe<kThroughput>(
                amplitude, cosine, sine, ix, iy, cx, cy, ra.xn, ra.yn);
            const double ixmiy = sub(ix, iy);
            double ap, pxp, pyp, an, pxn, pyn;
            lobe_rest(lp, cosine, sine, ix, iy, ixmiy, ap, pxp, pyp);
            lobe_rest(ln, cosine, sine, ix, iy, ixmiy, an, pxn, pyn);
            if (!have_b) {
                bar_sync(kBarB, (NWP + 1) * 32);
                have_b = true;
            }
            const RowB& rb = row_b[scalar_row];
            double q[4];
            {
                const double cxx = rb.cube[0], cyy = rb.cube[1];
                q[0] = divide_scaled(lp.wx_num, cxx, rb.rcp[0]);
                q[1] = divide_scaled(lp.wy_num, cyy, rb.rcp[1]);
                q[2] = divide_scaled(ln.wx_num, cxx, rb.rcp[0]);
                q[3] = divide_scaled(ln.wy_num, cyy, rb.rcp[1]);
                const unsigned e0 = exponent_field(lp.wx_num),
                               e1 = exponent_field(lp.wy_num);
                const unsigned e2 = exponent_field(ln.wx_num),
                               e3 = exponent_field(ln.wy_num);
                const unsigned lo = min(min(e0, e1), min(e2, e3)),
                               hi = max(max(e0, e1), max(e2, e3));
                if (!((rb.fast[0] & rb.fast[1]) && lo >= kScaledLo
                        && hi <= kScaledHi)) {
                    q[0] = __ddiv_rn(lp.wx_num, cxx);
                    q[1] = __ddiv_rn(lp.wy_num, cyy);
                    q[2] = __ddiv_rn(ln.wx_num, cxx);
                    q[3] = __ddiv_rn(ln.wy_num, cyy);
                }
            }
            const double sx = ra.sx, sy = ra.sy;
            double* pj = packed + (long long)k * 8 * M + pix;
            double* tj = transposed + ((long long)k * M + base) * 8;
            double2* tile = tiles + warp * 128;
            const int n = min(32, pix_hi - base);
            unsigned worst = 0;
            {
                // Plane 0: dipole difference.
                double v[8] = {sub(lp.unit, ln.unit),
                    mul(sub(q[0], q[2]), sx),
                    mul(sub(q[1], q[3]), sy),
                    sub(ap, an),
                    pxp,
                    pyp,
                    -pxn,
                    -pyn};
                emit<kThroughput>(
                    v, wc[0], pj, M, tj, tile, lane, n, active, vec_ok, worst);
            }
            if constexpr (P == 3) {
                // Plane 1: positive lobe.
                double v1[8] = {lp.unit,
                    mul(q[0], sx),
                    mul(q[1], sy),
                    ap,
                    pxp,
                    pyp,
                    0.0,
                    0.0};
                emit<kThroughput>(
                    v1,
                    wc[1],
                    pj + HW,
                    M,
                    tj + (long long)HW * 8,
                    tile,
                    lane,
                    n,
                    active,
                    vec_ok,
                    worst);
                // Plane 2: negative lobe.
                double v2[8] = {ln.unit,
                    mul(q[2], sx),
                    mul(q[3], sy),
                    an,
                    0.0,
                    0.0,
                    pxn,
                    pyn};
                emit<kThroughput>(
                    v2,
                    wc[2],
                    pj + 2 * HW,
                    M,
                    tj + 2 * (long long)HW * 8,
                    tile,
                    lane,
                    n,
                    active,
                    vec_ok,
                    worst);
            }
            if (active && worst >= 0x7ff00000u) {
                bad |= 1u << r;
            }
            r = nr;
            c = nc;
        }
        if (!have_b) {
            bar_sync(kBarB, (NWP + 1) * 32);
        }
        bad = __reduce_or_sync(0xffffffffu, bad);
        if (lane == 0 && bad) {
            atomicOr(&bad_rows, bad);
        }
    }

    __syncthreads();
    if (C == 1) {
        if (threadIdx.x < R) {
            finite_row[g + threadIdx.x * G] = !((bad_rows >> threadIdx.x) & 1u);
        }
        return;
    }
    cg::cluster_group cl = cg::this_cluster();
    cluster_wait();  // every CTA of the cluster has started (and initialized bad_rows)
    if (part != 0 && threadIdx.x == 0 && bad_rows) {
        atomicOr(cl.map_shared_rank(&bad_rows, 0), bad_rows);
    }
    cluster_arrive();
    cluster_wait();
    if (part == 0 && threadIdx.x < R) {
        finite_row[g + threadIdx.x * G] = !((bad_rows >> threadIdx.x) & 1u);
    }
}

// ---------------------------------------------------------------------------
// Launch geometry, one resident wave: G clusters of C CTAs, G <= K, rows dealt
// round-robin with R = ceil(K / G) <= 32 rows per cluster.
//  * Throughput mode (K >= #SMs): C = 1, six pixel warps per CTA, two CTAs per
//    SM, 128 registers. Chosen from B200 measurements: the large packets are
//    bound by the DRAM write pattern (a math-free skeleton of this kernel runs
//    as long as the kernel itself), and whole-row CTAs keep each row's J, J^T
//    and residual writes in a few long runs.
//  * Latency mode (K < #SMs): three helper warps per CTA; rows may be split
//    across a cluster. The cost model (microseconds) is
//      per_warp_items * kChunkLatency        dependent chain of the busiest warp
//      + per_sm_items * kChunkThroughput[P]  store/issue throughput of the busiest SM
//      + kClusterCost if C > 1
//    with residency (CTAs per SM, co-resident clusters) from the occupancy API.
constexpr int kVariants = 2;
// v0: latency mode (3 helper warps); v1: throughput mode, 128 registers.
// v2 uses the same latency geometry with direct preparation of at most 16 rows.
// v3: ten-warp, two-block throughput variant (96 registers), one plane only.
constexpr int kVariantThreads[kVariants] = {512, 512};
constexpr int kVariantHelpers[kVariants] = {kHelpers, 0};

using KernelFn = void (*)(
    const double*,
    const double*,
    const int*,
    const double*,
    double*,
    double*,
    double*,
    bool*,
    int,
    int,
    int,
    long long,
    int,
    int,
    unsigned,
    double,
    int);

template <int P>
KernelFn kernel_for(int variant) {
    if (variant == 2) {
        return jacobian_pack_kernel<P, 512, 1, false, true>;
    }
    if constexpr (P == 1) {
        if (variant == 3) {
            return jacobian_pack_kernel<P, 320, 2, true>;
        }
    }
    return variant == 0 ? jacobian_pack_kernel<P, 512, 1, false>
                        : jacobian_pack_kernel<P, 512, 1, true>;
}

struct Geometry {
    int C, NWP, G, variant, helpers;
};

// Per-warp J^T staging tile: 32 pixel records of 64 bytes.
// (throughput mode only; latency mode stores J^T records directly).
constexpr size_t tile_bytes(int NWP, int variant) {
    return (variant == 1 || variant == 3) ? (size_t)NWP * 32 * 64 : 0;
}

constexpr int kThroughputVariant = 1, kThroughputWarps = 6;
constexpr double kChunkLatency = 1.0;
constexpr double kChunkThroughput[4] = {0.0, 0.15, 0.0, 0.30};
constexpr double kClusterCost = 0.3;

template <int P>
cudaError_t choose_geometry(int K, int HW, int sms, Geometry& selected) {
    const int chunks = std::max(1, (HW + 31) / 32);
    if (K >= sms) {
        // Throughput mode (measured on B200): whole rows per CTA (no clusters: split
        // rows scatter the concurrent writes and cost up to 45% at 51x51), two
        // resident CTAs per SM of six pixel warps at 128 registers (no spills).
        // Rows are dealt round-robin, so the rows in flight form a dense band.
        // Long one-plane rows benefit from ten warps at a 96-register cap,
        // while short rows use four/six warps at 128 registers.
        const bool wide = P == 1 && chunks >= 64;
        const int variant = wide ? 3 : kThroughputVariant;
        const KernelFn fn = kernel_for<P>(variant);
        // Four-warps CTAs raise residency for large one-plane cohorts. Small
        // cohorts keep six warps because they cannot fill four CTAs per SM.
        const int NWP =
            wide ? 10 : (P == 1 && K >= 3 * sms ? 4 : kThroughputWarps);
        int occ = 0;
        const cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ, fn, 32 * NWP, tile_bytes(NWP, variant));
        if (error != cudaSuccess) {
            return error;
        }
        if (occ == 0) {
            occ = 1;
        }
        int G = (int)std::min<long long>(K, (long long)occ * sms);
        G = std::max(G, (K + kMaxRows - 1) / kMaxRows);
        selected = {1, NWP, G, variant, 0};
        return cudaSuccess;
    }
    // Latency mode: three helper warps per CTA; rows may be split across a
    // thread-block cluster so that every warp owns as few chunks as possible.
    Geometry best = {1, 1, K, 0, kHelpers};
    double best_cost = 1e30;
    const int v = 0;
    const KernelFn fn = kernel_for<P>(v);
    const int helpers = kVariantHelpers[v];
    for (int NWP = 1; NWP + helpers <= kVariantThreads[v] / 32; ++NWP) {
        const int T = 32 * (NWP + helpers);
        int occ = 0;
        const cudaError_t error = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ, fn, T, tile_bytes(NWP, v));
        if (error != cudaSuccess) {
            return error;
        }
        if (occ == 0) {
            continue;
        }
        for (int C = 1; C <= kMaxCluster && C <= chunks; C *= 2) {
            long long max_g = (long long)occ * sms / C;
            if (C > 1) {
                cudaLaunchConfig_t cfg = {};
                cfg.gridDim = dim3((unsigned)C);
                cfg.blockDim = dim3((unsigned)T);
                cfg.dynamicSmemBytes = tile_bytes(NWP, v);
                cudaLaunchAttribute attr[1];
                attr[0].id = cudaLaunchAttributeClusterDimension;
                attr[0].val.clusterDim.x = C;
                attr[0].val.clusterDim.y = 1;
                attr[0].val.clusterDim.z = 1;
                cfg.attrs = attr;
                cfg.numAttrs = 1;
                int clusters = 0;
                const cudaError_t cluster_error =
                    cudaOccupancyMaxActiveClusters(&clusters, fn, &cfg);
                if (cluster_error != cudaSuccess) {
                    return cluster_error;
                }
                max_g = std::min<long long>(max_g, clusters);
            }
            const int G = (int)std::min<long long>(K, max_g);
            if (G <= 0 || (K + G - 1) / G > kMaxRows) {
                continue;
            }
            const int R = (K + G - 1) / G;
            const int nch = (chunks + C - 1) / C;
            const long long items = (long long)R * nch;
            if (NWP > items) {
                continue;  // idle pixel warps
            }
            const long long per_warp = (items + NWP - 1) / NWP;
            const long long per_sm = ((long long)G * C + sms - 1) / sms * items;
            const double cost = per_warp * kChunkLatency
                + per_sm * kChunkThroughput[P] + (C > 1 ? kClusterCost : 0.0)
                + 0.001 * NWP;
            if (cost < best_cost) {
                best_cost = cost;
                best = {C, NWP, G, v, helpers};
            }
        }
    }
    selected = best;
    return cudaSuccess;
}

template <int P>
cudaError_t launch(
    const double* x,
    const double* weights,
    const int* indices,
    const double* residuals,
    double* packed,
    double* transposed,
    double* compact,
    bool* finite_row,
    long long N,
    int K,
    int H,
    int W,
    int device,
    cudaStream_t stream) {
    // Host-side launch configuration only (never results) is memoized: the SM
    // count per device and the geometry of the most recent shape on this thread.
    thread_local int sm_count[64] = {};
    int sms = device >= 0 && device < 64 ? sm_count[device] : 0;
    if (sms == 0) {
        const cudaError_t error = cudaDeviceGetAttribute(
            &sms, cudaDevAttrMultiProcessorCount, device);
        if (error != cudaSuccess) {
            return error;
        }
        if (device >= 0 && device < 64) {
            sm_count[device] = sms;
        }
    }
    if constexpr (P == 3) {
        // Register-capped CTAs with shared-memory input prefetch sustain more
        // resident warps for large cohorts of short three-plane rows.
        if (K >= 4 * sms && H * W >= 256 && H * W <= 512) {
            return launch_jacobian_pack_lean_fp64(
                x,
                weights,
                indices,
                residuals,
                packed,
                transposed,
                compact,
                finite_row,
                N,
                K,
                P,
                H,
                W,
                device,
                stream);
        }
    }
    struct Memo {
        int K = -1, HW = -1, device = -1;
        Geometry g;
    };
    thread_local Memo memo;
    if (memo.K != K || memo.HW != H * W || memo.device != device) {
        const cudaError_t error = choose_geometry<P>(K, H * W, sms, memo.g);
        if (error != cudaSuccess) {
            return error;
        }
        memo.K = K;
        memo.HW = H * W;
        memo.device = device;
    }
    Geometry g = memo.g;
    if (N <= 16) {
        g.variant = 2;
    }
    const int vec_ok = (reinterpret_cast<uintptr_t>(transposed) & 31) == 0;
    // pix / W == umulhi(pix, floor(2^32 / W) + 1) exactly when pix * W < 2^32 for
    // every pixel index (and W >= 2 so the multiplier fits); otherwise divide.
    const unsigned w_magic =
        (W >= 2 && (unsigned long long)H * W * W < (1ull << 32))
        ? (unsigned)((1ull << 32) / W + 1)
        : 0u;
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3((unsigned)(g.G * g.C));
    cfg.blockDim = dim3((unsigned)(32 * (g.NWP + g.helpers)));
    cfg.dynamicSmemBytes = tile_bytes(g.NWP, g.variant);
    cfg.stream = stream;
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = g.C;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr;
    cfg.numAttrs = g.C > 1 ? 1 : 0;
    // The native source passes the runtime cube exponent (3) as a kernel argument.
    const double cube_exponent = 3.0;
    return cudaLaunchKernelEx(
        &cfg,
        kernel_for<P>(g.variant),
        x,
        weights,
        indices,
        residuals,
        packed,
        transposed,
        compact,
        finite_row,
        K,
        H,
        W,
        N,
        g.G,
        g.C,
        w_magic,
        cube_exponent,
        vec_ok);
}

__global__ void empty_rows(bool* finite_row, long long K) {
    for (long long k = (long long)blockIdx.x * blockDim.x + threadIdx.x; k < K;
        k += (long long)gridDim.x * blockDim.x) {
        finite_row[k] = true;
    }
}

}  // namespace

cudaError_t launch_jacobian_pack_fp64(
    const double* x,
    const double* weights,
    const int* indices,
    const double* residuals,
    double* packed,
    double* transposed,
    double* compact,
    bool* finite_row,
    long long N,
    long long K,
    int P,
    int H,
    int W,
    int device,
    cudaStream_t stream) {
    if (K == 0) {
        return cudaSuccess;
    }
    if (H == 0 || W == 0) {
        empty_rows<<<
            (unsigned)std::min<long long>((K + 255) / 256, 65535),
            256,
            0,
            stream>>>(finite_row, K);
        return cudaGetLastError();
    }
    if (P == 1) {
        return launch<1>(
            x,
            weights,
            indices,
            residuals,
            packed,
            transposed,
            compact,
            finite_row,
            N,
            K,
            H,
            W,
            device,
            stream);
    }
    return launch<3>(
        x,
        weights,
        indices,
        residuals,
        packed,
        transposed,
        compact,
        finite_row,
        N,
        K,
        H,
        W,
        device,
        stream);
}

}  // namespace cuphoton::xfit::detail
