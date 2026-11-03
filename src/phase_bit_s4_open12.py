#!/usr/bin/env python3
"""
Paper 3, gravity in 3D: 1/r field and light deflection by a point mass
==============================================================================

A.  Field equation in 3D (lapse Phi = -U, N = exp(-U)): static Poisson
    nabla^2 Phi = kappa s on the cubic grid (FFT) for a compact source gives the
    1/r potential; the causal wave equation Phi_tt = c_g^2 (nabla^2 Phi - kappa s)
    switched on at t = 0 approaches the static field inside the light cone and
    does NOT grow (unlike 1D, where the retarded field grows ~ c t).
B.  Light deflection by a point mass, full 3D walker (2O*, eps = 0.3, mu = 0),
    static field U = GM / sqrt(r^2 + r_c^2), N = exp(-U):
      'W'   wait gate only (lapse N, gamma = 0):          deflection 2 GM / b
      'H2'  wait + stay gate, lambda(N) compensated (A = 1/N, gamma = 1): 4 GM / b
    A massless packet (upper Weyl doublet, k0 along x) passes the mass at impact
    parameter b; the deflection is the change of <k_y>, compared with
      - the straight-ray integral  n_s k0 * int dU/dy dx  over the travelled path,
      - the density-averaged local force  n_s N^n_s c' <|k|> dU/dy  at each tick (the straight
        ray is only a guide: the packet is as wide as the impact parameter),
    n_s = 1 (W) or 2 (H2).  Both Kramers partners averaged.

Run parts: python phase_bit_s4_open12.py [A] [B]   (default: both)
"""

import functools
import os
import sys
import time

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_fcc_3d import DT_HALF, FCC_STEPS, LAT_A
from phase_bit_mass_s4 import bands_C
from phase_bit_s4_open import CP
from phase_bit_s4_open4 import a_of
from phase_bit_s4_open9 import Xl, LAM0, lam_of
from paths import data_path

print = functools.partial(print, flush=True)
A_FIT = np.load(data_path('phase_bit_s4_open9.npz'))['A_fit']


# ═════════════════════════════════════════════════════════════════════════════
def part_A(out):
    print("A. 3D field equation: static Poisson (1/r) and causal wave equation (no secular growth)")
    n = 96
    h = LAT_A
    x = (np.arange(n) - n // 2) * h
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    R = np.sqrt(X ** 2 + Y ** 2 + Z ** 2)
    sig = 2.0
    s = np.exp(-R ** 2 / (2 * sig ** 2))
    s /= s.sum() * h ** 3                       # total "load" 1 (per volume)
    kap = 1.0
    # static: periodic FFT Poisson with the discrete 7-point Laplacian, mean removed
    k = 2 * np.pi * np.fft.fftfreq(n, d=h)
    KX, KY, KZ = np.meshgrid(k, k, k, indexing='ij')
    lapk = -(4 / h ** 2) * (np.sin(KX * h / 2) ** 2 + np.sin(KY * h / 2) ** 2 + np.sin(KZ * h / 2) ** 2)
    sk = np.fft.fftn(s - s.mean())
    lapk[0, 0, 0] = 1.0
    Phik = kap * sk / lapk
    Phik[0, 0, 0] = 0.0
    Phi = np.real(np.fft.ifftn(Phik))
    # compare radial profile with -kappa Q /(4 pi r) (plus a constant from the periodic box)
    rs = np.array([4, 6, 8, 10, 12, 16, 20]) * 1.0
    prof = np.array([Phi[(np.abs(R - r) < 0.4)].mean() for r in rs])
    ref = -kap / (4 * np.pi * rs)
    off = np.mean(prof - ref)
    rows = []
    for r_, p_, f_ in zip(rs, prof, ref):
        rows.append((r_, p_ - off, f_))
        print(f"   r = {r_:5.1f}: Phi = {p_-off:+.5f}, -kappa/(4 pi r) = {f_:+.5f}, ratio {(p_-off)/f_:.4f}")
    out['A_static'] = np.array(rows)
    # gradient (force) is free of the constant: dPhi/dr vs kappa/(4 pi r^2)
    g = np.gradient(Phi, h, axis=0)[n // 2:, n // 2, n // 2]
    xr = x[n // 2:]
    for r_ in (6.0, 10.0, 16.0):
        i = np.argmin(np.abs(xr - r_))
        print(f"      dPhi/dr at r = {xr[i]:.2f}: {g[i]:.6f} vs kappa/(4 pi r^2) = {kap/(4*np.pi*xr[i]**2):.6f}")
    # causal: leapfrog wave equation, source switched on at t = 0, field 0, open (Dirichlet) box
    cg = CP
    dt = 0.5 * h / cg / np.sqrt(3)                # Courant 0.5/sqrt3 per axis
    f, fo = np.zeros_like(Phi), np.zeros_like(Phi)
    c2 = (cg * dt) ** 2
    ii = n // 2
    rec = []
    tmax = 26.0 / cg
    nsteps = int(tmax / dt)
    probes = [6, 10, 16]
    for it in range(nsteps):
        L = np.zeros_like(f)
        L[1:-1, 1:-1, 1:-1] = (f[2:, 1:-1, 1:-1] + f[:-2, 1:-1, 1:-1] + f[1:-1, 2:, 1:-1] + f[1:-1, :-2, 1:-1]
                               + f[1:-1, 1:-1, 2:] + f[1:-1, 1:-1, :-2] - 6 * f[1:-1, 1:-1, 1:-1]) / h ** 2
        fn = 2 * f - fo + c2 * (L - kap * s)
        fo, f = f, fn
        if it % 10 == 0:
            t = (it + 1) * dt
            gr = np.gradient(f[:, ii, ii], h)
            rec.append((t, *[gr[ii + int(round(p / h))] for p in probes], f[ii, ii, ii]))
    rec = np.array(rec)
    out['A_wave'] = rec
    for j, p in enumerate(probes):
        late = rec[-1, 1 + j]
        print(f"   causal: dPhi/dr at r = {p}: onset ~ r/c_g = {p/cg:.1f}; at t = {rec[-1,0]:.1f}: {late:.6f} "
              f"(static {kap/(4*np.pi*p**2):.6f})")
    cen = rec[:, -1]
    print(f"   causal: Phi at the centre at t = {rec[len(rec)//2,0]:.1f} / {rec[-1,0]:.1f}: {cen[len(rec)//2]:.5f} / {cen[-1]:.5f} "
          f"(static, free space: {-kap/(4*np.pi)*np.sqrt(2/np.pi)/sig:.5f}) -> settles, no growth ~ t")


# ═════════════════════════════════════════════════════════════════════════════
class Walker3D:
    """3D H2 / W scheme on a periodic cubic box (FCC = even sublattice), registers of shape (S, 24)."""

    def __init__(self, shape, kind, Nf, mu=0.0, nbins=48):
        self.shape, self.kind = shape, kind
        self.S = int(np.prod(shape))
        Nflat = Nf.ravel()
        a = a_of(Nflat)
        self.cg, self.sg = np.cos(a).astype(np.float32)[:, None], np.sin(a).astype(np.float32)[:, None]
        if kind == 'W':
            self.bins = [(np.arange(self.S), expm(1j * Xl(LAM0)(mu)).astype(np.complex64))]
        else:
            edges = np.linspace(Nflat.min() - 1e-9, 1.0 + 1e-9, nbins + 1)
            b = np.digitize(Nflat, edges) - 1
            self.bins = []
            for i in range(nbins):
                idx = np.where(b == i)[0]
                if len(idx) == 0:
                    continue
                Nc = 0.5 * (edges[i] + edges[i + 1])
                lam = LAM0 if Nc > 0.999 else lam_of(max(Nc, 0.4), A_FIT)
                self.bins.append((idx, expm(1j * Xl(lam)(mu)).astype(np.complex64)))

    def coin(self, v):
        out = np.empty_like(v)
        for idx, C in self.bins:
            out[idx] = v[idx] @ C.T
        return out

    def shift(self, v):
        w = v.reshape(*self.shape, 12, 2)
        out = np.empty_like(w)
        for d in range(12):
            out[..., d, :] = np.roll(w[..., d, :], tuple(FCC_STEPS[d]), axis=(0, 1, 2))
        return out.reshape(self.S, 24)

    def tick(self, st):
        cg, sg = self.cg, self.sg
        m1 = self.shift(self.coin(st[0]))
        if self.kind == 'W':                      # wait gate with N (registers: m, w)
            w = -st[1]
            return [cg * m1 + 1j * sg * w, 1j * sg * m1 + cg * w]
        s1 = -self.coin(st[1])                    # stay register: coin, no shift
        a_ = cg * m1 + 1j * sg * s1               # stay gate with N_h = N
        b_ = 1j * sg * m1 + cg * s1
        del m1, s1
        wa, wb = -st[2], -st[3]                   # wait gate with N_w = N on both
        return [cg * a_ + 1j * sg * wa, cg * b_ + 1j * sg * wb, 1j * sg * a_ + cg * wa, 1j * sg * b_ + cg * wb]


def packet3d(shape, k0, sig, x0, member):
    """massless packet on the upper Weyl doublet (2O*, mu = 0), Gaussian envelope of width sig, at x0."""
    C = expm(1j * Xl(LAM0)(0.0))
    ks = [2 * np.pi * np.fft.fftfreq(n, d=LAT_A) for n in shape]
    KX, KY, KZ = np.meshgrid(*ks, indexing='ij')
    g = np.exp(-sig ** 2 * ((KX - k0[0]) ** 2 + (KY - k0[1]) ** 2 + (KZ - k0[2]) ** 2) / 2)
    E0, V0 = bands_C(np.asarray(k0, float), C)
    up = lambda E, kk: np.argsort(np.abs(E - CP * kk))[:2]
    Q0, _ = np.linalg.qr(V0[:, up(E0, np.linalg.norm(k0))])
    uref = Q0[:, member]
    psik = np.zeros((*shape, 24), np.complex64)
    for i in zip(*np.where(g > 1e-6)):
        k = np.array([KX[i], KY[i], KZ[i]])
        E, V = bands_C(k, C)
        Q, _ = np.linalg.qr(V[:, up(E, np.linalg.norm(k))])
        u = Q @ (Q.conj().T @ uref)
        psik[i] = g[i] * u / max(np.linalg.norm(u), 1e-12)
    # position x0 via phase, then to real space; keep the FCC (even) sublattice
    xs = [(np.arange(n) - n // 2) * LAT_A for n in shape]
    psi = np.fft.ifftn(psik * np.exp(-1j * (KX * x0[0] + KY * x0[1] + KZ * x0[2]))[..., None], axes=(0, 1, 2))
    # fftfreq grid places x = 0 at index 0: roll so that index n//2 is x = 0
    psi = np.roll(psi, tuple(n // 2 for n in shape), axis=(0, 1, 2))
    I = np.indices(shape).sum(0) - sum(n // 2 for n in shape)
    psi *= (I % 2 == 0)[..., None]
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2))
    return psi.reshape(-1, 24).astype(np.complex64)


def part_B(out):
    print("B. light deflection by a point mass in 3D (2O*, eps 0.3, massless packet)")
    t0 = time.time()
    shape = (112, 88, 80)
    k0, sig = 0.6, 5.0
    GM, rc = 0.25, 2.0
    nst, every = 128, 4
    xs = [(np.arange(n) - n // 2) * LAT_A for n in shape]
    X, Y, Z = np.meshgrid(*xs, indexing='ij')
    ymass = -14 * LAT_A                           # mass below the grid centre, beam near y = 0 ... b
    R = np.sqrt(X ** 2 + (Y - ymass) ** 2 + Z ** 2)
    U = GM / np.sqrt(R ** 2 + rc ** 2)
    Nf = np.exp(-U)
    dUdy = -GM * (Y - ymass) / (R ** 2 + rc ** 2) ** 1.5
    S = int(np.prod(shape))
    print(f"   box {shape} = {S} sites, {S*24*8/1e9:.2f} GB per register (complex64); N_min = {Nf.min():.4f}")
    ks = [2 * np.pi * np.fft.fftfreq(n, d=LAT_A) for n in shape]
    KX, KY, KZ = np.meshgrid(*ks, indexing='ij')
    near = (KX - k0) ** 2 + KY ** 2 + KZ ** 2 < 2.0 ** 2      # the physical copy (not the sublattice image)
    KABS = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    res = {}
    for b_idx in (12, 20):
        b = b_idx * LAT_A
        x0 = np.array([-30 * LAT_A, ymass + b, 0.0])
        for kind, ns in (('W', 1), ('H2', 2)):
            wk = Walker3D(shape, 'W' if kind == 'W' else 'WH', Nf)
            nreg = 2 if kind == 'W' else 4
            gfield = ns * Nf.ravel() ** ns * CP * dUdy.ravel()    # local force per unit k for a massless packet
            Kys, Fps, Xs, ts = [], [], [], []
            for member in (0, 1):
                psi = packet3d(shape, (k0, 0.0, 0.0), sig, x0, member)
                st = [psi] + [np.zeros_like(psi) for _ in range(nreg - 1)]
                # filter against the pi branches of the gates
                ph = np.exp(1j * CP * k0 * Nf.ravel()[np.argmin(np.abs(X.ravel() - x0[0]) + np.abs(Y.ravel() - x0[1])
                                                                 + np.abs(Z.ravel()))] ** ns * DT_HALF)
                for _ in range(3):
                    st2 = wk.tick(st)
                    st = [(a + ph * c) / 2 for a, c in zip(st, st2)]
                    del st2
                nrm = np.sqrt(sum(np.vdot(a, a).real for a in st))
                st = [a / nrm for a in st]
                ky, fp, xc, tt = [], [], [], []
                for it in range(nst + 1):
                    if it % every == 0:
                        rho = sum((np.abs(a) ** 2).sum(1) for a in st)
                        spec = sum((np.abs(np.fft.fftn(a.reshape(*shape, 24), axes=(0, 1, 2))) ** 2).sum(-1) for a in st)
                        sw = spec * near
                        ky.append((sw * KY).sum() / sw.sum())
                        kabs = (sw * KABS).sum() / sw.sum()      # massless: the force scales with |k|
                        fp.append(kabs * (rho @ gfield) / rho.sum())
                        xc.append(((rho @ X.ravel()) / rho.sum(), (rho @ Y.ravel()) / rho.sum()))
                        tt.append(it * DT_HALF)
                    if it < nst:
                        st = wk.tick(st)
                Kys.append(ky); Fps.append(fp); Xs.append(xc); ts = tt
                print(f"      b = {b_idx}, {kind:2s}, member {member}: k_y {ky[0]:+.5f} -> {ky[-1]:+.5f}, "
                      f"x {xc[0][0]:+.1f} -> {xc[-1][0]:+.1f}  [{time.time()-t0:.0f}s]")
                del st
            ky = np.mean(Kys, 0); fp = np.mean(Fps, 0); xc = np.mean(Xs, 0); ts = np.array(ts)
            dky = ky - ky[0]
            # local-force prediction integrated in time; straight-ray prediction over the travelled x range
            pred_loc = np.concatenate([[0], np.cumsum(0.5 * (fp[1:] + fp[:-1]) * np.diff(ts))])
            xa, xb = xc[0, 0], xc[-1, 0]
            bb = x0[1] - ymass
            ray = lambda xx: -GM * bb * xx / ((bb ** 2 + rc ** 2) * np.sqrt(xx ** 2 + bb ** 2 + rc ** 2))
            pred_ray = ns * k0 * (ray(xb) - ray(xa))
            full = -ns * k0 * 2 * GM * bb / (bb ** 2 + rc ** 2)
            res[(b_idx, kind)] = (dky[-1], pred_loc[-1], pred_ray, full)
            out[f'B_{b_idx}_{kind}'] = np.array([ts, dky, pred_loc, xc[:, 0], xc[:, 1]])
            print(f"   b = {b_idx} ({bb:.2f}), {kind}: delta k_y = {dky[-1]:+.5f}; local force {pred_loc[-1]:+.5f} "
                  f"(ratio {dky[-1]/pred_loc[-1]:.3f}); straight ray over x {xa:+.1f} .. {xb:+.1f}: {pred_ray:+.5f} "
                  f"(ratio {dky[-1]/pred_ray:.3f}); full path {full:+.5f} -> angle {dky[-1]/k0:+.4f} rad")
        r = res[(b_idx, 'H2')][0] / res[(b_idx, 'W')][0]
        print(f"   b = {b_idx}: deflection H2 / W = {r:.3f}  (GR: 2 = (1 + gamma) with gamma = 1)")
    out['B_summary'] = np.array([[b, k, *(res[(b, k)])] for b in (12, 20) for k in ('W', 'H2')], dtype=object)
    out['B_par'] = np.array([k0, sig, GM, rc, nst])


def main():
    parts = [p.upper() for p in sys.argv[1:]] or ['A', 'B']
    path = data_path('phase_bit_s4_open12.npz')
    out = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}
    for p in parts:
        {'A': part_A, 'B': part_B}[p](out)
        np.savez_compressed(path, **out)
    print("saved data/phase_bit_s4_open12.npz")


if __name__ == '__main__':
    main()
