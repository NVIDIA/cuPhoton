// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// SPDX-License-Identifier: Apache-2.0 AND MIT
//
// CUDA implementation of the structured root solver.
// One CUDA thread per polynomial; native FP64; original numerical gates.
// Adapted mathematical routines: jverzani/AMRVW.jl, commit
// 8cb7afc20abd9dff4216380faa3a149cce16421f, src/{transformations,companion,
// RDS,chains,rfactorization,qfactorization,diagonal-block,create-bulge,
// AMRVW_algorithm,bulge-step,utils}.jl.
// Rotation convention is [c s; -s c]; indices remain one-based.
// The original MIT notice applies to adapted portions; the Apache notice
// above applies to new validation/error-handling and CUDA transport code.
//
// The MIT License (MIT)
// Copyright (c) 2019 John Verzani
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// NVRTC supplies CUDA math declarations; do not require a host math.h.

// Required compile options: --std=c++17 --fmad=false. No fast-math.
// Explicit fma below must remain fused, matching the qualified CPU source.
#define D __device__
struct Rot {
    double c, s;
    int i;
};
struct Triple {
    Rot a, b, c;
};
struct Givens {
    double c, s, r;
};
struct Block {
    double a, b, c, d;
};
struct Roots2 {
    double ar, ai, br, bi;
};
struct Scaled {
    double value;
    int exponent;
};
enum Code {
    converged = 0,
    iteration_limit = 1,
    invalid_input = 2,
    bounds = 3,
    numerical_range = 4,
    identity_failure = 5,
    running = 6
};

struct Lane {
    const double* input;
    double *state, *roots_real, *roots_imag;
    int degree, batch, lane, n, error = 0, error_site = 0, count = 0,
                                sweeps = 0, deflations = 0, exceptional = 0;
    D bool need(bool condition, int code, int site) {
        if (!condition && !error) {
            error = code;
            error_site = site;
        }
        return !error;
    }
    D double finite(double x) {
        return need(isfinite(x), numerical_range, 1) ? x : 0;
    }
    D double ratio(double a, double b) {
        if (!need(b != 0, numerical_range, 2)) {
            return 0;
        }
        double x = finite(a / b);
        return need(a == 0 || x != 0, numerical_range, 3) ? x : 0;
    }
    D double coeff(int i) {
        if (!need(i >= 0 && i <= degree, bounds, 4)) {
            return 0;
        }
        return input[(long long)i * batch + lane];
    }
    D Rot identity(int i = 1) {
        return {1, 0, i};
    }
    D Rot checked(Rot u) {
        finite(u.c);
        finite(u.s);
        if (!need(u.i >= 1, bounds, 5)) {
            return identity();
        }
        if (!need(
                fabs(u.c * u.c + u.s * u.s - 1) < 2e-12, numerical_range, 6)) {
            return identity();
        }
        return u;
    }
    D Rot get(int chain, int i) {
        if (!need(
                chain >= 0 && chain < 3 && i >= 1 && i <= n
                    && (chain != 0 || i < n),
                bounds,
                7)) {
            return identity();
        }
        long long base =
            ((long long)(2 * chain) * (degree + 1) + i) * batch + lane;
        return {state[base], state[base + (long long)(degree + 1) * batch], i};
    }
    D void put(int chain, int i, Rot u) {
        if (!need(
                chain >= 0 && chain < 3 && i >= 1 && i <= n
                    && (chain != 0 || i < n) && u.i == i,
                bounds,
                8)) {
            return;
        }
        long long base =
            ((long long)(2 * chain) * (degree + 1) + i) * batch + lane;
        state[base] = u.c;
        state[base + (long long)(degree + 1) * batch] = u.s;
    }
    D void push(double real, double imag = 0) {
        if (!need(count < degree, bounds, 9)) {
            return;
        }
        roots_real[(long long)count * batch + lane] = real;
        roots_imag[(long long)count * batch + lane] = imag;
        ++count;
    }
    D Rot adj(Rot u) {
        return {u.c, -u.s, u.i};
    }
    D Givens givens(double a, double b) {
        finite(a);
        finite(b);
        if (error) {
            return {1, 0, 0};
        }
        double r = finite(hypot(a, b));
        if (error || r == 0) {
            return {1, 0, 0};
        }
        return {a / r, b / r, r};
    }
    D Rot polish(double a, double b, int i) {
        double delta = finite(a * a + b * b - 1);
        if (!need(fabs(delta) < 1e-8, numerical_range, 10)) {
            return identity(i);
        }
        double scale = 1 - delta / 2;
        return checked({a * scale, b * scale, i});
    }
    D Rot fuse(Rot a, Rot b) {
        checked(a);
        checked(b);
        if (!need(a.i == b.i, bounds, 11)) {
            return identity(a.i);
        }
        return polish(a.c * b.c - a.s * b.s, a.c * b.s + a.s * b.c, a.i);
    }
    D Triple turnover(Rot u, Rot v, Rot w) {
        checked(u);
        checked(v);
        checked(w);
        if (!need(
                u.i == w.i && (u.i - v.i == 1 || v.i - u.i == 1), bounds, 12)) {
            return {identity(), identity(), identity()};
        }
        double a = -v.c * w.s * u.c - w.c * u.s, b = v.s * w.s;
        if (a == 0 && b == 0 && v.s * w.c == 0) {
            return {checked({1, 0, v.i}),
                checked({u.c * w.c - v.c * u.s * w.s, 0, u.i}),
                checked({v.c, 0, v.i})};
        }
        Givens g = givens(a, b);
        double c4 = g.c, s4 = -g.s;
        Rot r5 = polish(u.c * w.c - v.c * u.s * w.s, g.r, u.i);
        r5.s = -r5.s;
        if (error) {
            return {identity(), identity(), identity()};
        }
        Rot r6;
        if (r5.s != 0) {
            r6 = polish(u.c * v.s * s4 + v.c * c4, u.s * v.s / r5.s, v.i);
        } else {
            double c6 = u.c * v.s * s4 + v.c * c4;
            double s6 = r5.c * (u.c * v.s * c4 - v.c * s4);
            r6 = polish(c6, s6, v.i);
        }
        return {checked({c4, s4, v.i}), r5, r6};
    }
    D void initialize() {
        double parity = n % 2 ? -1 : 1;
        // Independent coefficient normalization is loaded on demand instead
        // of materializing the CPU xs vector. Dependent Givens order is intact.
        Givens g = givens(parity * ratio(coeff(n), coeff(0)), -1);
        if (error) {
            return;
        }
        put(1, n, checked({g.c, -g.s, n}));
        put(2, n, checked({g.s, -g.c, n}));
        for (int i = n - 1; i >= 1 && !error; --i) {
            double x =
                parity * (i % 2 ? 1 : -1) * ratio(coeff(n - i), coeff(0));
            g = givens(x, g.r);
            if (error) {
                return;
            }
            put(1, i, checked({g.c, -g.s, i}));
            put(2, i, checked({g.c, g.s, i}));
            put(0, i, {0, 1, i});
        }
    }
    D double w(int j, int k) {
        if (error) {
            return 0;
        }
        if (j == k) {
            return -get(2, k).s;
        }
        double prod = get(2, j).c * get(2, k).c;
        for (int i = j + 1; i < k && !error; ++i) {
            prod *= get(2, i).s;
        }
        return prod;
    }
    D double r(int j, int k) {
        if (error || j <= 0 || k < j) {
            return 0;
        }
        if (!need(k <= n, bounds, 13)) {
            return 0;
        }
        double prod = get(1, j).s;
        if (!need(prod != 0, numerical_range, 14)) {
            return 0;
        }
        double total = finite(w(j, k) / prod), par = -1;
        for (int i = j + 1; i <= k && !error; ++i) {
            prod *= get(1, i).s;
            if (!need(prod != 0, numerical_range, 15)) {
                return 0;
            }
            total = finite(
                total + par * get(1, j).c * get(1, i).c / prod * w(i, k));
            par = -par;
        }
        return total;
    }
    D double qentry(int i, int j) {
        if (error || i <= 0 || j <= 0 || i > j + 1) {
            return 0;
        }
        if (!need(i <= n && j <= n, bounds, 16)) {
            return 0;
        }
        if (i == j + 1) {
            return -get(0, j).s;
        }
        double prod = i > 1 ? get(0, i - 1).c : 1;
        if (j < n) {
            prod *= get(0, j).c;
        }
        for (int k = i; k < j && !error; ++k) {
            prod *= get(0, k).s;
        }
        return prod;
    }
    D Block block(int j) {
        if (!need(j >= 1 && j < n, bounds, 17)) {
            return {0, 0, 0, 0};
        }
        int k = j + 1;
        return {finite(qentry(j, j - 1) * r(j - 1, j) + qentry(j, j) * r(j, j)),
            finite(
                qentry(j, j - 1) * r(j - 1, k) + qentry(j, j) * r(j, k)
                + qentry(j, k) * r(k, k)),
            finite(qentry(k, j) * r(j, j)),
            finite(qentry(k, j) * r(j, k) + qentry(k, k) * r(k, k))};
    }
    D Rot pass_desc(int chain, Rot u, int last) {
        int i = u.i;
        if (!need(i >= 1 && i < last, bounds, 18)) {
            return identity();
        }
        Triple out = turnover(get(chain, i), get(chain, i + 1), u);
        if (error) {
            return identity();
        }
        put(chain, i, out.b);
        put(chain, i + 1, out.c);
        return out.a;
    }
    D Rot pass_r(Rot u) {
        u = pass_desc(2, u, n);
        int i = u.i;
        if (!need(i > 1 && i <= n, bounds, 19)) {
            return identity();
        }
        Triple out = turnover(get(1, i), get(1, i - 1), u);
        if (error) {
            return identity();
        }
        put(1, i, out.b);
        put(1, i - 1, out.c);
        return out.a;
    }
    D double parity(int k) {
        if (k == 0 || k == n) {
            return 1;
        }
        if (!need(k > 0 && k < n && get(0, k).s == 0, bounds, 20)) {
            return 1;
        }
        return copysign(1.0, get(0, k).c);
    }
    D Rot flip(Rot u, double sign) {
        u.s *= sign;
        return checked(u);
    }
    D void sweep(int lo, int hi, unsigned kick) {
        if (!need(lo >= 1 && hi <= n && hi - lo >= 2, bounds, 21)) {
            return;
        }
        double left_sign = parity(lo - 1), right_sign = parity(hi);
        Rot u, v;
        if (error) {
            return;
        }
        if (kick) {
            unsigned word = kick * 2654435761u + unsigned(lo) * 2246822519u
                + unsigned(hi) * 3266489917u;
            double angle =
                (double(word) + 0.5) * (3.14159265358979323846 / 4294967296.0);
            u = checked({sin(angle), cos(angle), lo});
            v = checked({u.c, -u.s, lo + 1});
        } else {
            Block tail = block(hi - 1), top = block(lo), next = block(lo + 1);
            double scale = 0;
            scale = fmax(scale, fabs(tail.a));
            scale = fmax(scale, fabs(tail.b));
            scale = fmax(scale, fabs(tail.c));
            scale = fmax(scale, fabs(tail.d));
            scale = fmax(scale, fabs(top.a));
            scale = fmax(scale, fabs(top.b));
            scale = fmax(scale, fabs(top.c));
            scale = fmax(scale, fabs(top.d));
            scale = fmax(scale, fabs(next.c));
            if (!need(scale > 0, numerical_range, 22)) {
                return;
            }
            tail.a /= scale;
            tail.b /= scale;
            tail.c /= scale;
            tail.d /= scale;
            top.a /= scale;
            top.b /= scale;
            top.c /= scale;
            top.d /= scale;
            double tr = tail.a + tail.d,
                   det = tail.a * tail.d - tail.b * tail.c;
            double x = finite(top.a * top.a + top.b * top.c - tr * top.a + det);
            double y = finite(top.c * (top.a + top.d - tr));
            double z = finite(top.c * (next.c / scale));
            Givens gv = givens(y, z);
            v = checked({gv.c, -gv.s, lo + 1});
            Givens gu = givens(x, gv.r);
            u = checked({gu.c, -gu.s, lo});
        }
        if (error) {
            return;
        }
        Triple left = turnover(adj(u), adj(v), flip(get(0, lo), left_sign));
        Rot limb = left.a;
        put(0, lo, flip(left.b, left_sign));
        put(0, lo + 1, fuse(left.c, get(0, lo + 1)));
        for (int step = 0; step < hi - lo + 1 && !error; ++step) {
            v = pass_r(v);
            u = pass_r(u);
            if (error) {
                return;
            }
            if (v.i < hi - 1) {
                v = pass_desc(0, v, n - 1);
                u = pass_desc(0, u, n - 1);
                if (error) {
                    return;
                }
                Triple moved = turnover(limb, v, u);
                v = moved.a;
                u = moved.b;
                limb = moved.c;
            } else {
                if (!need(v.i == hi - 1, bounds, 23)) {
                    return;
                }
                put(0, v.i, fuse(get(0, v.i), flip(v, right_sign)));
                u = pass_desc(0, u, n - 1);
                u = fuse(limb, u);
                u = pass_r(u);
                put(0, u.i, fuse(get(0, u.i), flip(u, right_sign)));
                return;
            }
        }
        need(false, bounds, 24);
    }
    D Scaled determinant(double a, double d, double b, double c) {
        int ea = 0, ed = 0, eb = 0, ec = 0;
        double ma = frexp(a, &ea), md = frexp(d, &ed), mb = frexp(b, &eb),
               mc = frexp(c, &ec);
        int ad = (a == 0 || d == 0) ? -8192 : ea + ed,
            bc = (b == 0 || c == 0) ? -8192 : eb + ec;
        int exponent = ad > bc ? ad : bc;
        if (exponent == -8192) {
            return {0, 0};
        }
        double p0 = ma * md, q0 = mb * mc;
        double p = scalbn(p0, ad - exponent), q = scalbn(q0, bc - exponent);
        double pe = scalbn(fma(ma, md, -p0), ad - exponent),
               qe = scalbn(fma(mb, mc, -q0), bc - exponent);
        double h = p - q, z = h - p, low = (p - (h - z)) + (-q - z) + pe - qe;
        return {h + low, exponent};
    }
    D Roots2 eigen2(double a, double b, double c, double d) {
        finite(a);
        finite(b);
        finite(c);
        finite(d);
        if (error) {
            return {0, 0, 0, 0};
        }
        if (b == 0 || c == 0) {
            return {a, 0, d, 0};
        }
        bool same_sign = signbit(a) == signbit(d);
        double center, delta;
        if (same_sign) {
            center = a + (d - a) / 2;
            delta = (a - d) / 2;
        } else {
            double ha = ratio(a, 2), hd = ratio(d, 2);
            center = ha + hd;
            delta = ha - hd;
        }
        if (!need(
                (a == -d || center != 0) && (a == d || delta != 0),
                numerical_range,
                25)) {
            return {0, 0, 0, 0};
        }
        int edelta = 0, eb = 0, ec = 0;
        frexp(delta, &edelta);
        double mb = frexp(b, &eb), mc = frexp(c, &ec);
        int sum = eb + ec, half = sum >= 0 ? (sum + 1) / 2 : sum / 2;
        int de = delta == 0 ? -8192 : edelta,
            be = (b == 0 || c == 0) ? -8192 : half;
        int exponent = de > be ? de : be;
        if (exponent == -8192) {
            return {center, 0, center, 0};
        }
        double v = scalbn(delta, -exponent), p = v * v, pe = fma(v, v, -p),
               q0 = mb * mc;
        int shift = sum - 2 * exponent;
        double q = (b == 0 || c == 0) ? 0 : scalbn(q0, shift);
        double qe = (b == 0 || c == 0) ? 0 : scalbn(fma(mb, mc, -q0), shift);
        double h = p + q, z = h - p,
               discriminant = h + ((p - (h - z)) + (q - z) + pe + qe);
        double width = finite(scalbn(sqrt(fabs(discriminant)), exponent));
        if (!need(discriminant == 0 || width != 0, numerical_range, 26)) {
            return {0, 0, 0, 0};
        }
        if (discriminant < 0) {
            return {center, width, center, -width};
        }
        if (width == 0) {
            return {center, 0, center, 0};
        }
        double large = finite(center + copysign(width, center));
        if (error || large == 0) {
            return {0, 0, 0, 0};
        }
        Scaled det = determinant(a, d, b, c);
        int er = 0;
        double mr = frexp(large, &er);
        double small = finite(scalbn(det.value / mr, det.exponent - er));
        if (!need(det.value == 0 || small != 0, numerical_range, 27)) {
            return {0, 0, 0, 0};
        }
        return {large, 0, small, 0};
    }
    D int solve(int cap, int& unconverged) {
        if (!need(
                degree >= 1 && degree <= 4096 && cap >= -1,
                invalid_input,
                28)) {
            return error;
        }
        if (!need(coeff(0) != 0, invalid_input, 29)) {
            return error;
        }
        for (int i = 0; i <= degree; ++i) {
            if (!need(isfinite(coeff(i)), invalid_input, 30)) {
                return error;
            }
        }
        n = degree;
        while (n > 0 && coeff(n) == 0) {
            push(0);
            --n;
        }
        if (error) {
            return error;
        }
        if (n == 1) {
            push(-ratio(coeff(1), coeff(0)));
        }
        if (n == 2) {
            Roots2 e = eigen2(
                0, -ratio(coeff(2), coeff(0)), 1, -ratio(coeff(1), coeff(0)));
            push(e.ar, e.ai);
            push(e.br, e.bi);
        }
        if (error) {
            return error;
        }
        if (n <= 2) {
            return need(count == degree, identity_failure, 31) ? converged
                                                               : error;
        }
        initialize();
        if (error) {
            return error;
        }
        if (cap < 0) {
            cap = 20 * n;
        }
        int hi = n, previous_lo = -1, previous_hi = -1, since = 0;
        while (hi > 0 && !error) {
            if (hi == 1) {
                push(finite(qentry(1, 1) * r(1, 1)));
                --hi;
                break;
            }
            int lo = 1;
            for (int k = hi - 1; k >= 1 && !error; --k) {
                Rot q = get(0, k);
                if (fabs(q.s) <= 0x1p-52) {
                    if (q.s != 0) {
                        ++deflations;
                    }
                    put(0, k, {copysign(1.0, q.c), 0, k});
                    lo = k + 1;
                    break;
                }
            }
            if (error) {
                return error;
            }
            if (lo != previous_lo || hi != previous_hi) {
                since = 0;
            }
            previous_lo = lo;
            previous_hi = hi;
            if (hi - lo == 0) {
                Block b = block(hi - 1);
                push(b.d);
                --hi;
                continue;
            }
            if (hi - lo == 1) {
                Block b = block(lo);
                Roots2 e = eigen2(b.a, b.b, b.c, b.d);
                push(e.ar, e.ai);
                push(e.br, e.bi);
                hi = lo - 1;
                continue;
            }
            if (sweeps >= cap) {
                unconverged = hi;
                return iteration_limit;
            }
            unsigned kick = 0;
            if (since >= 15) {
                kick = unsigned(++exceptional);
                since = 0;
            }
            sweep(lo, hi, kick);
            if (error) {
                return error;
            }
            ++sweeps;
            ++since;
        }
        if (error) {
            return error;
        }
        return need(count == degree, identity_failure, 32) ? converged : error;
    }
};

// Coefficients: descending polynomial (degree+1,batch), NOT LP recurrence c.
// state: (6,degree+1,batch), chains q.c,q.s,ct.c,ct.s,b.c,b.s.
// Roots: two (degree,batch) arrays, appended in the original solver order.
// stats: (8,batch) int32: status,count,unconverged,sweeps,deflations,kicks,
// first_error_site,effective_degree. Nonconverged/failed rows are never usable.
// Unused output slots start as NaN. No warp collectives, atomics or barriers.
extern "C" __global__ void rds_lane(
    const double* input,
    double* state,
    double* roots_real,
    double* roots_imag,
    int* stats,
    int degree,
    int batch,
    int max_sweeps) {
    int lane = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (lane >= batch) {
        return;
    }
    for (int k = 0; k < 8; ++k) {
        stats[(long long)k * batch + lane] = 0;
    }
    stats[lane] = running;
    if (degree < 1 || degree > 4096) {
        stats[lane] = invalid_input;
        return;
    }
    for (int i = 0; i < degree; ++i) {
        roots_real[(long long)i * batch + lane] = nan("");
        roots_imag[(long long)i * batch + lane] = nan("");
    }
    Lane x{input, state, roots_real, roots_imag, degree, batch, lane, degree};
    int unconverged = 0, code = x.solve(max_sweeps, unconverged);
    stats[lane] = code;
    stats[(long long)batch + lane] = x.count;
    stats[2LL * batch + lane] = unconverged;
    stats[3LL * batch + lane] = x.sweeps;
    stats[4LL * batch + lane] = x.deflations;
    stats[5LL * batch + lane] = x.exceptional;
    stats[6LL * batch + lane] = x.error_site;
    stats[7LL * batch + lane] = x.n;
}
