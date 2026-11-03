#!/usr/bin/env python3
"""
Paper 3, Experiment 3: isotropy of the cuboctahedral shells and of the walker's light-cone front
==============================================================================

The original "Bloch sphere" idea: the nodes of the cuboctahedral shell n of the FCC
lattice (graph distance n, 10 n^2 + 2 nodes) as a discrete sphere.  How isotropic
are they, and how isotropic is what the walker actually does?

Anisotropy power of a weighted direction set (w_i, r_i):
    C_l = 4 pi/(2l+1) * sum_m |a_lm|^2,   a_lm = sum_i w_i Y_lm(r_i) / sum_i w_i
C_l = 0 for an isotropic distribution, 1 for a single point.  Any O_h-symmetric set
has C_1 = C_2 = C_3 = 0 (it is a spherical 3-design), so l = 4, 6, 8 are the tests.

A.  Geometry: shell n nodes projected to the unit sphere, n = 1 ... 200, and the
    continuum limit (cuboctahedron surface, uniform area density, projected).
B.  Walker, k space (2O*, eps = 0.3, counterterm): long-time direction distribution
    of a point source with all 24 internal states (incoherent) = directions of the
    group velocities of all bands over the BZ; Dirac quartet alone; smooth sources of
    width sigma (weight exp(-sigma^2 k^2)) -> C_l(sigma); front speed max_k v.n per
    direction.
C.  Walker, real space: point source, heading d0 with both spins, symmetrised over
    the 24 cube rotations (= incoherent sum over all 24 internal states), n up to 64.

D.  Front speed per direction from the simulated distribution (99 % / 99.9 % radius in a cone).

Run parts: python phase_bit_exp3_isotropy.py [A] [B] [C] [D]   (default: all)
"""

import functools
import itertools
import os
import sys
import time

import numpy as np
from scipy.linalg import expm
from scipy.special import sph_harm_y

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_fcc_3d import MOVES_PHYS, DT_HALF, FCC_STEPS, LAT_A, step3
from phase_bit_s4_open import CP
from phase_bit_s4_open4 import model_2Ostar
from paths import data_path

print = functools.partial(print, flush=True)
LS = (4, 6, 8)


def C_l(vecs, w=None, ls=LS):
    """anisotropy power C_l of directions vecs (n,3) with weights w."""
    v = np.asarray(vecs, float)
    r = np.linalg.norm(v, axis=1)
    keep = r > 0
    v, r = v[keep], r[keep]
    w = np.ones(len(v)) if w is None else np.asarray(w, float)[keep]
    th = np.arccos(np.clip(v[:, 2] / r, -1, 1))
    ph = np.arctan2(v[:, 1], v[:, 0])
    out = []
    for l in ls:
        s = 0.0
        for m in range(-l, l + 1):
            a = np.sum(w * sph_harm_y(l, m, th, ph)) / w.sum()
            s += abs(a) ** 2
        out.append(4 * np.pi / (2 * l + 1) * s)
    return np.array(out)


def rotations24():
    """the 24 proper cube rotations as (perm, signs) acting on index axes."""
    rots = []
    for p in itertools.permutations(range(3)):
        for s in itertools.product((1, -1), repeat=3):
            M = np.zeros((3, 3), int)
            for i in range(3):
                M[i, p[i]] = s[i]
            if round(np.linalg.det(M)) == 1:
                rots.append((p, s))
    return rots


# ═════════════════════════════════════════════════════════════════════════════
def part_A(out):
    print("A. cuboctahedral shells of the FCC lattice projected to the unit sphere")
    t0 = time.time()
    rows = []
    for n in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 200):
        r = np.arange(-n, n + 1)
        X, Y, Z = np.meshgrid(r, r, r, indexing='ij')
        A_ = np.abs(X) + np.abs(Y) + np.abs(Z)
        par = (X + Y + Z) % 2 == 0
        dist = np.maximum(np.maximum(np.maximum(np.abs(X), np.abs(Y)), np.abs(Z)), A_ // 2)
        sel = par & (dist == n)
        pts = np.stack([X[sel], Y[sel], Z[sel]], 1)
        cl = C_l(pts)
        rows.append((n, len(pts), *cl))
        print(f"   n = {n:4d}: {len(pts):7d} nodes (10n^2+2 = {10*n*n+2}); C_4 = {cl[0]:.3e}, C_6 = {cl[1]:.3e}, "
              f"C_8 = {cl[2]:.3e}  [{time.time()-t0:.0f}s]")
    out['A_rows'] = np.array(rows)
    # continuum limit: cuboctahedron surface (8 triangles in {111} planes, 6 squares in {100} planes),
    # sampled with the FCC node density of each face type: 1/sqrt3 on {111}, 1/2 on {100} (index units)
    rng = np.random.default_rng(1)
    tri = []
    for s in itertools.product((1, -1), repeat=3):
        s = np.array(s)
        V = [s * np.array(v) for v in ((1, 1, 0), (1, 0, 1), (0, 1, 1))]
        tri.append(V)
    sq = []
    for ax in range(3):
        for sg in (1, -1):
            V = []
            for a, b in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                v = np.zeros(3)
                v[ax] = sg
                o = [i for i in range(3) if i != ax]
                v[o[0]], v[o[1]] = a, b
                V.append(v)
            sq.append(V)
    A_tri, A_sq = np.sqrt(3) / 4 * 2, 2.0          # areas: triangle side sqrt2, square side sqrt2
    nT, nS = 8 * A_tri / np.sqrt(3), 6 * A_sq / 2   # expected node numbers per face type
    M = 2_000_000
    k_tri = rng.random(M) < nT / (nT + nS)
    u, v = rng.random(M), rng.random(M)
    flip = u + v > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    fi = rng.integers(0, 8, M)
    T = np.array(tri, float)
    p_t = T[fi, 0] + u[:, None] * (T[fi, 1] - T[fi, 0]) + v[:, None] * (T[fi, 2] - T[fi, 0])
    fj = rng.integers(0, 6, M)
    S = np.array(sq, float)
    a, b = rng.random(M), rng.random(M)
    p_s = S[fj, 0] + a[:, None] * (S[fj, 1] - S[fj, 0]) + b[:, None] * (S[fj, 3] - S[fj, 0])
    pts = np.where(k_tri[:, None], p_t, p_s)
    cl = C_l(pts)
    out['A_limit'] = cl
    print(f"   continuum limit (cuboctahedron surface, uniform, projected): C_4 = {cl[0]:.3e}, C_6 = {cl[1]:.3e}, "
          f"C_8 = {cl[2]:.3e}")
    # reference: a single O_h orbit of generic directions and N random points
    for N in (12, 1000, 100000):
        print(f"   reference: {N} random points: C_4 ~ {C_l(rng.normal(size=(N, 3)))[0]:.1e} (expect ~1/N = {1/N:.1e})")


# ═════════════════════════════════════════════════════════════════════════════
def bz_velocities(C, K):
    """group velocities (Hellmann-Feynman, exact) and energies of all 24 bands at the k points K (n, 3)."""
    vel, Es, kk = [], [], []
    W = np.repeat(MOVES_PHYS, 2, axis=0) / DT_HALF        # (24, 3) velocity of each internal state
    for chunk in np.array_split(K, max(1, len(K) // 4000)):
        ph = np.exp(-1j * chunk @ MOVES_PHYS.T)          # (n, 12)
        U = np.repeat(ph, 2, axis=1)[:, :, None] * C[None]
        lam, V = np.linalg.eig(U)
        E = -np.angle(lam) / DT_HALF
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        p = np.abs(V) ** 2                                # (n, 24 comps, 24 bands)
        vel.append(np.einsum('nib,ix->nbx', p, W))
        Es.append(E)
        kk.append(np.repeat(chunk[:, None, :], 24, axis=1))
    return np.concatenate(vel).reshape(-1, 3), np.concatenate(Es).ravel(), np.concatenate(kk).reshape(-1, 3)


def part_B(out):
    print("B. walker in k space (2O*, eps = 0.3): long-time direction distribution")
    t0 = time.time()
    X, lam = model_2Ostar()
    rows = []
    rng = np.random.default_rng(7)

    def quartet_mask(E):
        Ek = E.reshape(-1, 24)
        idx = np.argsort(np.abs(Ek), axis=1)[:, :4]
        m_ = np.zeros_like(Ek, bool)
        np.put_along_axis(m_, idx, True, axis=1)
        return m_.ravel()
    for mu in (0.0, 0.03):
        C = expm(1j * X(mu))
        # point source: uniform random k in the cube [-pi/a, pi/a)^3 (covers the BZ exactly twice)
        Kc = rng.uniform(-np.pi / LAT_A, np.pi / LAT_A, (120000, 3))
        v, E, K = bz_velocities(C, Kc)
        cl_all = C_l(v)
        # identify the quartet per k: the 4 eigenvalues closest to 0
        mask = quartet_mask(E)
        cl_q = C_l(v[mask])
        print(f"   mu = {mu}: point source, all 24 bands: C_4 = {cl_all[0]:.3e}, C_6 = {cl_all[1]:.3e}, C_8 = {cl_all[2]:.3e}"
              f"; Dirac quartet only: {cl_q[0]:.3e}, {cl_q[1]:.3e}, {cl_q[2]:.3e}  [{time.time()-t0:.0f}s]")
        out[f'B_point_{mu}'] = np.array([*cl_all, *cl_q])
        # smooth sources of width sigma: |psi(k)|^2 ~ exp(-sigma^2 k^2); deterministic quadrature:
        # Fibonacci directions x radial Gauss-Legendre nodes with weight k^2 exp(-sigma^2 k^2)
        nd = 3000
        i_ = np.arange(nd) + 0.5
        zz = 1 - 2 * i_ / nd
        az = np.pi * (1 + np.sqrt(5)) * i_
        dirs = np.stack([np.sqrt(1 - zz ** 2) * np.cos(az), np.sqrt(1 - zz ** 2) * np.sin(az), zz], 1)
        floor = C_l(dirs)
        print(f"      quadrature floor (C_l of the k directions themselves): {floor[0]:.1e}, {floor[1]:.1e}")
        xg, wg = np.polynomial.legendre.leggauss(16)
        for sig in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
            kr = (xg + 1) / 2 * 3.5 / sig
            wr = wg * kr ** 2 * np.exp(-sig ** 2 * kr ** 2)
            Kg = (kr[:, None, None] * dirs[None]).reshape(-1, 3)
            wk = np.repeat(wr, nd)
            vg, Eg, _ = bz_velocities(C, Kg)
            mg = quartet_mask(Eg)
            wb = np.repeat(wk, 24)
            ca = C_l(vg, wb)
            cq = C_l(vg[mg], wb[mg])
            rows.append((mu, sig, *ca, *cq))
            print(f"      sigma = {sig:5.1f}: all bands C_4 = {ca[0]:.3e}, C_6 = {ca[1]:.3e}; quartet C_4 = {cq[0]:.3e}, "
                  f"C_6 = {cq[1]:.3e}, C_8 = {cq[2]:.3e}")
        out[f'B_floor_{mu}'] = floor
        # front speed: max over k and bands of v . n (uniform BZ sample)
        fr = []
        for nvec in ((1, 0, 0), (1, 1, 0), (1, 1, 1)):
            nv = np.array(nvec, float) / np.linalg.norm(nvec)
            fr.append(((v @ nv).max(), (v[mask] @ nv).max()))
        fr = np.array(fr)
        out[f'B_front_{mu}'] = fr
        print(f"   front speed max v.n [100]/[110]/[111]: all bands {np.round(fr[:,0],4)}, quartet {np.round(fr[:,1],4)}; "
              f"c = {np.sqrt(3):.4f}, c' = {CP:.4f}")
    out['B_rows'] = np.array(rows)
    # small-k Dirac cone isotropy (group speed of the quartet vs direction)
    C = expm(1j * X(0.0))
    from phase_bit_mass_s4 import bands_C
    rr = []
    for k in (0.05, 0.1, 0.2, 0.4, 0.8):
        sp = []
        for nvec in ((1, 0, 0), (1, 1, 0), (1, 1, 1)):
            nv = np.array(nvec, float) / np.linalg.norm(nvec)
            Ep = bands_C(nv * (k + 1e-4), C)[0]
            Em = bands_C(nv * (k - 1e-4), C)[0]
            q = lambda E_: np.sort(E_[np.argsort(np.abs(E_))[:4]])[-1]
            sp.append((q(Ep) - q(Em)) / 2e-4)
        rr.append((k, *sp))
        print(f"   massless cone, upper branch: |v|(k = {k}) along [100]/[110]/[111] = {np.round(sp, 4)} "
              f"(spread {(max(sp)-min(sp))/np.mean(sp):.2e})")
    out['B_cone'] = np.array(rr)


# ═════════════════════════════════════════════════════════════════════════════
def part_C(out, nmax=64):
    print(f"C. real space: point source, 2O* (eps 0.3, mu 0.03), n up to {nmax}")
    t0 = time.time()
    X, lam = model_2Ostar()
    C = expm(1j * X(0.03)).astype(np.complex64)
    N = 2 * nmax + 3
    c = N // 2
    ix = np.arange(N) - c
    IX, IY, IZ = np.meshgrid(ix, ix, ix, indexing='ij')
    dist = np.maximum(np.maximum(np.maximum(np.abs(IX), np.abs(IY)), np.abs(IZ)),
                      (np.abs(IX) + np.abs(IY) + np.abs(IZ)) // 2)
    pts = np.stack([IX.ravel(), IY.ravel(), IZ.ravel()], 1)
    d0 = 0
    rots = rotations24()
    checkpoints = (8, 16, 24, 32, 48, 64)
    P = {n: np.zeros((N, N, N), np.float64) for n in checkpoints if n <= nmax}
    print(f"   box {N}^3, {N**3 * 24 * 8 / 1e9:.2f} GB per state (complex64)")
    for spin in (0, 1):
        psi = np.zeros((N, N, N, 12, 2), np.complex64)
        psi[c, c, c, d0, spin] = 1.0
        for n in range(1, nmax + 1):
            psi = step3(psi, C, 2)
            if n in P:
                P[n] += (np.abs(psi) ** 2).sum(axis=(3, 4))
                print(f"      spin {spin}, n = {n}: norm {np.sum(np.abs(psi)**2):.6f}  [{time.time()-t0:.0f}s]")
        del psi
    rows = []
    for n, Pn in P.items():
        # symmetrise over the 24 cube rotations (incoherent sum over all 24 internal states)
        Ps = np.zeros_like(Pn)
        for p, s in rots:
            A = np.transpose(Pn, p)
            for ax in range(3):
                if s[ax] < 0:
                    A = np.flip(A, ax)
            Ps += A
        Ps /= Ps.sum()
        w = Ps.ravel()
        cl_all = C_l(pts, w)
        front = (dist.ravel() >= 0.8 * n)
        cl_f = C_l(pts[front], w[front])
        r = np.linalg.norm(pts, axis=1) * LAT_A
        t = n * DT_HALF
        hist, edges = np.histogram(r / t, bins=40, range=(0, 2.0), weights=w)
        rows.append((n, *cl_all, *cl_f, w[front].sum()))
        out[f'C_rad_{n}'] = hist
        # direction-resolved radial weight along [100], [110], [111] cones
        print(f"   n = {n}: all C_4 = {cl_all[0]:.3e}, C_6 = {cl_all[1]:.3e}, C_8 = {cl_all[2]:.3e}; "
              f"front (dist >= 0.8 n, weight {w[front].sum():.3f}): C_4 = {cl_f[0]:.3e}, C_6 = {cl_f[1]:.3e}")
        if n == max(P):
            out['C_P_last'] = Ps.astype(np.float32)
    out['C_rows'] = np.array(rows)
    out['C_rad_edges'] = edges


def part_D(out):
    """front of the simulated distribution (n = last checkpoint) per direction: radius below which
    99 % / 99.9 % of the probability in a 12-degree cone around [100], [110], [111] lies, over t."""
    print("D. simulated front speed per direction (from the stored distribution)")
    Ps = out['C_P_last'].astype(np.float64)
    n = int(out['C_rows'][-1, 0])
    N = Ps.shape[0]
    c = N // 2
    ix = (np.arange(N) - c) * LAT_A
    IX, IY, IZ = np.meshgrid(ix, ix, ix, indexing='ij')
    R = np.sqrt(IX ** 2 + IY ** 2 + IZ ** 2)
    t = n * DT_HALF
    rows = []
    for nvec in ((1, 0, 0), (1, 1, 0), (1, 1, 1)):
        nv = np.array(nvec, float) / np.linalg.norm(nvec)
        cosang = (IX * nv[0] + IY * nv[1] + IZ * nv[2]) / np.maximum(R, 1e-12)
        cone = (cosang > np.cos(np.radians(12))) & (R > 0)
        r, w = R[cone], Ps[cone]
        o = np.argsort(r)
        cw = np.cumsum(w[o]) / w.sum()
        r99, r999 = r[o][np.searchsorted(cw, 0.99)], r[o][np.searchsorted(cw, 0.999)]
        rows.append((r99 / t, r999 / t))
        print(f"   {nvec}: 99 % within r/t = {r99/t:.3f}, 99.9 % within {r999/t:.3f}  (n = {n}, t = {t})")
    out['D_front'] = np.array(rows)


def main():
    parts = [p.upper() for p in sys.argv[1:]] or ['A', 'B', 'C', 'D']
    path = data_path('phase_bit_exp3_isotropy.npz')
    out = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}
    for p in parts:
        {'A': part_A, 'B': part_B, 'C': part_C, 'D': part_D}[p](out)
        np.savez_compressed(path, **out)
    print("saved data/phase_bit_exp3_isotropy.npz")


if __name__ == '__main__':
    main()
