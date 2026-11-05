#!/usr/bin/env python3
"""
Paper 3 follow-up, gravitational waves 3: what the lattice field equations radiate
==============================================================================

The model's fields are linear combinations of the trace-reversed perturbation hbar_mu_nu
of harmonic-gauge linearised GR, box hbar_mu_nu = -16 pi T_mu_nu:
    lapse   h_00 = (hbar_00 + hbar_kk)/2        source T_00 + T_kk      (round 6)
    drag    h_0i = hbar_0i                       source T_0i             (g0i, C = 4)
    space   published model:  h_ij = h_00 delta_ij               (A = 1/N)
            with bond field:  h_ij = h_00 delta_ij + H_ij,  H_ij = hbar_ij - delta_ij hbar_kk
                              (= GR: hbar_ij + delta_ij (hbar_00 - hbar_kk)/2)
All ten hbar components are evolved with the same lattice wave equation (7-point Laplacian,
leapfrog, c = 1, h = 1, dt = 0.5, absorbing sponge), source = a conserved rotating m = 2
pattern, single frequency Omega, compact and conserved identically (superpotential):
    Psi^{0i0j} = A_ij(t) g(x),  A = Re[a e^{-iWt}],  a = (e_x + i e_y)(e_x + i e_y)^T,  g Gaussian
    T^00 = A_ij d_i d_j g,  T^0j = -dA_ij/dt d_i g,  T^ij = d^2 A_ij/dt^2 g
(A first version with T^0i = grad chi and a spectrally solved stress had 1/r^3 tails that
reach the sponge; the far field was then not 1/r.)
After the transient the complex amplitude of every component is locked in over two
periods.  Gauge-invariant analysis in the wave zone: the electric Riemann tensor
    E_ij = R_0i0j = -1/2 (h_ij,00 + h_00,ij - h_0i,0j - h_0j,0i)
decomposed in the frame (e_theta, e_phi, n): plus, cross (tensor), breathing (transverse
trace), longitudinal (nn), vector (n e).  GR: only plus and cross at order 1/r.
Also: harmonic condition, TT amplitude vs the retarded far-field formula
    hbar_ij = (4/r) e^{i W r} T_ij^(k = W n),  angular pattern (1 + cos^2)/2 and cos.

    python phase_bit_s4_open19.py reanalyze <lock.npz>   (spectral re-analysis of saved lock-in fields)
    python phase_bit_s4_open19.py        (~10 min, ~1.3 GB; lock-in fields go to the scratch
                                          path given by GW3_SCRATCH, default: not saved)
"""

import functools
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import data_path

print = functools.partial(print, flush=True)
NG, H, DT = 160, 1.0, 0.5
NPER = 32                                # steps per period
W = 2 * np.pi / (NPER * DT)              # Omega = 0.3927, wavelength 16
S_SRC = 2.0
SPONGE, G0 = 32, 0.25                    # sponge = 2 wavelengths, cubic profile
COMP = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)]
CI = {c: i for i, c in enumerate(COMP)}


def idx(m, n):
    return CI[(min(m, n), max(m, n))]


def source():
    """compact conserved source from a superpotential Psi^{0i0j} = A_ij(t) g(x):
    T^00 = A_ij d_i d_j g, T^0j = -dA_ij/dt d_i g, T^ij = d^2A_ij/dt^2 g  (conserved identically);
    A_ij = Re[a_ij e^{-iWt}], a = (e_x + i e_y)(e_x + i e_y)^T (rotating m = 2), g Gaussian of width s."""
    x = (np.arange(NG) - NG // 2) * H
    X, Y, Z = np.meshgrid(x, x, x, indexing='ij')
    Xs = (X, Y, Z)
    g = np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * S_SRC ** 2))
    e = np.array([1.0, 1j, 0.0])
    a = np.outer(e, e)
    dg = [-Xi / S_SRC ** 2 * g for Xi in Xs]
    T = {}
    T[(0, 0)] = sum(a[i, j] * (Xs[i] * Xs[j] / S_SRC ** 4 - (i == j) / S_SRC ** 2) * g
                    for i in range(3) for j in range(3) if a[i, j] != 0)
    for j in range(3):
        T0j_up = sum(1j * W * a[i, j] * dg[i] for i in range(3) if a[i, j] != 0)   # -dA/dt -> +iW a
        T[(0, j + 1)] = -T0j_up if np.ndim(T0j_up) else np.zeros_like(g, complex)  # T_0j = -T^0j
    for i in range(3):
        for j in range(i, 3):
            T[(i + 1, j + 1)] = -W ** 2 * a[i, j] * g if a[i, j] != 0 else np.zeros_like(g, complex)
    # conservation check (spectral): d^mu T_mu nu = -d_t T_0nu + d_j T_jnu,  d_t -> -iW
    k = 2 * np.pi * np.fft.fftfreq(NG, d=H)
    K = np.meshgrid(k, k, k, indexing='ij')
    err = []
    for nu in range(4):
        Tn = T[(0, nu)] if nu else T[(0, 0)]
        dv = 1j * W * np.fft.fftn(Tn) + sum(1j * K[j] * np.fft.fftn(T[tuple(sorted((j + 1, nu)))]) for j in range(3))
        err.append(np.abs(dv).max() / (W * np.abs(np.fft.fftn(Tn)).max() + 1e-30))
    return T, (X, Y, Z), err


def lap(f):
    return (np.roll(f, 1, 0) + np.roll(f, -1, 0) + np.roll(f, 1, 1) + np.roll(f, -1, 1)
            + np.roll(f, 1, 2) + np.roll(f, -1, 2) - 6 * f) / H ** 2


def sponge():
    i = np.arange(NG)
    d = np.minimum(i, NG - 1 - i).astype(np.float32)
    g1 = np.where(d < SPONGE, G0 * ((SPONGE - d) / SPONGE) ** 3, 0).astype(np.float32)
    return np.maximum(np.maximum(g1[:, None, None], g1[None, :, None]), g1[None, None, :])


def evolve(T):
    srcR = [(16 * np.pi * T[c].real).astype(np.float32) for c in COMP]
    srcI = [(16 * np.pi * T[c].imag).astype(np.float32) for c in COMP]
    gam = sponge()
    a, b = (1 - gam * DT / 2), 1 / (1 + gam * DT / 2)
    del gam
    h = [np.zeros((NG,) * 3, np.float32) for _ in COMP]
    ho = [np.zeros((NG,) * 3, np.float32) for _ in COMP]
    lock = [np.zeros((NG,) * 3, np.complex64) for _ in COMP]
    ramp, start = 2 * NPER, 10 * NPER
    nsteps = start + 2 * NPER
    t0 = time.time()
    for it in range(nsteps):
        t = it * DT
        env = 0.5 * (1 - np.cos(np.pi * min(t / (ramp * DT), 1.0)))
        cs, sn = np.float32(env * np.cos(W * t)), np.float32(env * np.sin(W * t))
        for c in range(len(COMP)):
            # source Re[A e^{-iWt}] = ReA cos + ImA sin
            new = b * (2 * h[c] - a * ho[c] + DT ** 2 * (lap(h[c]) + cs * srcR[c] + sn * srcI[c]))
            ho[c], h[c] = h[c], new.astype(np.float32)
        if it >= start:
            ph = np.complex64(np.exp(1j * W * (it + 1) * DT) * DT * 2 / (2 * NPER * DT))
            for c in range(len(COMP)):
                lock[c] += h[c] * ph
        if it % 64 == 0:
            print(f"   step {it}/{nsteps}  max|hbar_00| {np.abs(h[0]).max():.3e}  [{time.time()-t0:.0f}s]")
    return lock


def trilin(F, p):
    g = p / H + NG // 2
    i0 = np.floor(g).astype(int)
    f = g - i0
    v = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[0] if dx else 1 - f[0]) * (f[1] if dy else 1 - f[1]) * (f[2] if dz else 1 - f[2])
                v = v + w * F[i0[0] + dx, i0[1] + dy, i0[2] + dz]
    return v


def metric_at(lock, p, model):
    """h_mu nu (4x4 complex) at point p for model 'GR' (with bond field) or 'pub' (A = 1/N)."""
    hb = np.zeros((4, 4), complex)
    for m in range(4):
        for n in range(4):
            hb[m, n] = trilin(lock[idx(m, n)], p)
    tr = hb[1, 1] + hb[2, 2] + hb[3, 3]
    h = np.zeros((4, 4), complex)
    h[0, 0] = (hb[0, 0] + tr) / 2
    h[0, 1:] = h[1:, 0] = hb[0, 1:]
    if model == 'GR':
        h[1:, 1:] = hb[1:, 1:] + np.eye(3) * (hb[0, 0] - tr) / 2
    else:
        h[1:, 1:] = np.eye(3) * h[0, 0]
    return h, hb


def riemann_E(lock, p, model):
    e = np.eye(3) * H
    hp = lambda q: metric_at(lock, q, model)[0]
    h0 = hp(p)
    E = np.zeros((3, 3), complex)
    # first derivatives of h_0i and second derivatives of h_00 (central differences)
    d1 = [(hp(p + e[j])[0, 1:] - hp(p - e[j])[0, 1:]) / (2 * H) for j in range(3)]   # d1[j][i] = h_0i,j
    for i in range(3):
        for j in range(3):
            if i == j:
                h00ij = (hp(p + e[i])[0, 0] - 2 * h0[0, 0] + hp(p - e[i])[0, 0]) / H ** 2
            else:
                h00ij = (hp(p + e[i] + e[j])[0, 0] - hp(p + e[i] - e[j])[0, 0]
                         - hp(p - e[i] + e[j])[0, 0] + hp(p - e[i] - e[j])[0, 0]) / (4 * H ** 2)
            E[i, j] = -0.5 * (-W ** 2 * h0[i + 1, j + 1] + h00ij + 1j * W * d1[j][i] + 1j * W * d1[i][j])
    return E


def frame(n):
    th = np.arccos(np.clip(n[2], -1, 1))
    ph = np.arctan2(n[1], n[0])
    et = np.array([np.cos(th) * np.cos(ph), np.cos(th) * np.sin(ph), -np.sin(th)])
    ep = np.array([-np.sin(ph), np.cos(ph), 0.0])
    return et, ep, th, ph


def decompose(E, n):
    et, ep, *_ = frame(n)
    P = lambda a, b: a @ E @ b
    return dict(plus=(P(et, et) - P(ep, ep)) / 2, cross=P(et, ep), breath=(P(et, et) + P(ep, ep)) / 2,
                long=P(n, n), vec=np.hypot(abs(P(n, et)), abs(P(n, ep))))


def that(T, XYZ, k):
    """continuous Fourier transform of the compact source at wave vector k (|x| < 14)."""
    X, Y, Z = XYZ
    m = (np.abs(X) < 14) & (np.abs(Y) < 14) & (np.abs(Z) < 14)
    ph = np.exp(-1j * (k[0] * X[m] + k[1] * Y[m] + k[2] * Z[m])) * H ** 3
    return {c: (T[c][m] * ph).sum() for c in COMP}


def main():
    out = {}
    print(f"GW3: ten hbar components on {NG}^3, Omega = {W:.4f} (wavelength {2*np.pi/W:.1f}), source width {S_SRC}")
    T, XYZ, err = source()
    print("   source conservation |d^mu T_mu nu| / |d_t T_0 nu| (spectral): " + ", ".join(f"{e:.1e}" for e in err))
    lock = evolve(T)
    scr = os.environ.get('GW3_SCRATCH')
    if scr:
        np.savez(scr, *lock)
    # harmonic condition at sample points: i W hbar_0nu + d_j hbar_j nu
    dirs = []
    for th in (0.0, 30.0, 60.0, 90.0):
        for ph in (0.0, 20.0, 45.0):
            if th == 0 and ph > 0:
                continue
            t, p = np.radians(th), np.radians(ph)
            dirs.append(np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)]))
    rows = []
    for r in (20.0, 30.0, 40.0):
        for n in dirs:
            p = r * n
            _, hb = metric_at(lock, p, 'GR')
            harm = []
            for nu in range(4):
                dj = sum((trilin(lock[idx(j + 1, nu)], p + H * np.eye(3)[j]) -
                          trilin(lock[idx(j + 1, nu)], p - H * np.eye(3)[j])) / (2 * H) for j in range(3))
                harm.append(abs(1j * W * hb[0, nu] + dj) / (W * np.abs(hb).max()))
            res = {}
            for model in ('GR', 'pub'):
                res[model] = decompose(riemann_E(lock, p, model), n)
            # retarded far field prediction for the TT part of hbar (plus) and the measured TT of hbar
            et, ep, th, ph = frame(n)
            Th = that(T, XYZ, W * n)
            Tij = np.array([[Th[(min(i, j) + 1, max(i, j) + 1)] for j in range(3)] for i in range(3)])
            pred_plus = 4 / r * np.exp(1j * W * r) * (et @ Tij @ et - ep @ Tij @ ep) / 2
            pred_cross = 4 / r * np.exp(1j * W * r) * (et @ Tij @ ep)
            hs = hb[1:, 1:]
            meas_plus, meas_cross = (et @ hs @ et - ep @ hs @ ep) / 2, et @ hs @ ep
            g, pb = res['GR'], res['pub']
            tt = np.hypot(abs(g['plus']), abs(g['cross']))
            rows.append([r, th, ph, max(harm), abs(g['plus']), abs(g['cross']), abs(g['breath']) / tt,
                         abs(g['long']) / tt, g['vec'] / tt, abs(pb['plus']) / tt, abs(pb['cross']) / tt,
                         abs(pb['breath']) / tt, abs(pb['long']) / tt, pb['vec'] / tt,
                         abs(meas_plus), abs(pred_plus), abs(meas_cross), abs(pred_cross),
                         abs(g['plus']) / (W ** 2 / 2 * abs(meas_plus) + 1e-30)])
    R = np.array(rows)
    out['rows'] = R
    out['cols'] = np.array(['r', 'theta', 'phi', 'harm', 'GR_plus', 'GR_cross', 'GR_breath/TT', 'GR_long/TT',
                            'GR_vec/TT', 'pub_plus/TT', 'pub_cross/TT', 'pub_breath/TT', 'pub_long/TT',
                            'pub_vec/TT', 'hbar_plus', 'pred_plus', 'hbar_cross', 'pred_cross', 'E/(W^2 h/2)'])
    print("\n   r    theta  phi | harm   | GR: plus     cross    breath/TT long/TT vec/TT | published: plus/TT cross/TT "
          "breath/TT long/TT vec/TT | hbar_+ / pred, hbar_x / pred")
    for x in R:
        print(f"   {x[0]:4.0f} {np.degrees(x[1]):5.0f} {np.degrees(x[2]):4.0f} | {x[3]:.1e} | {x[4]:.3e} {x[5]:.3e} "
              f"{x[6]:.1e}   {x[7]:.1e}  {x[8]:.1e} | {x[9]:.1e} {x[10]:.1e} {x[11]:.3f} {x[12]:.3f} {x[13]:.3f} | "
              f"{x[14]/x[15]:.4f} " + (f"{x[16]/x[17]:.4f}" if x[17] > 1e-3 * x[15] else "  (no x)"))
    # angular pattern of the GR tensor amplitude at r = 45, phi = 0 (normalised to theta = 0)
    sel = (R[:, 0] == 30.0) & (R[:, 2] == 0.0)
    q = R[sel]
    print("\n   pattern at r = 30: theta, |plus|/|plus(0)| vs (1 + cos^2)/2, |cross|/|plus(0)| vs cos")
    for x in q:
        print(f"     {np.degrees(x[1]):4.0f}: {x[4]/q[0,4]:.4f} ({(1+np.cos(x[1])**2)/2:.4f}),  {x[5]/q[0,4]:.4f} ({np.cos(x[1]):.4f})")
    # 1/r scaling
    for th in (0.0, np.radians(60)):
        a = [R[(R[:, 0] == r) & (np.abs(R[:, 1] - th) < 1e-9) & (R[:, 2] == 0)][0] for r in (20.0, 30.0, 40.0)]
        print(f"   theta {np.degrees(th):.0f}: r*|GR plus| at r = 20/30/40: " + ", ".join(f"{x[0]*x[4]:.4e}" for x in a)
              + "; published breath/TT: " + ", ".join(f"{x[11]:.3f}" for x in a)
              + "; GR (breath, long)/TT: " + ", ".join(f"({x[6]:.1e}, {x[7]:.1e})" for x in a))
    out['conservation'] = np.array(err)
    analyze_spectral(lock, T, XYZ, out)
    np.savez(data_path('phase_bit_s4_open19.npz'), **out)




# ═════════════════════════════════════════════════════════════════════════════
# re-analysis with spectral derivatives on grid points (no interpolation, no finite differences)
def spectral_fields(lock, native):
    """E_ij (6 grids) for both models and the harmonic residual (4 grids).
    native = False: d_j -> i k_j, d_t -> -i W.
    native = True : the lattice's own symbols, d_j -> (2i/h) sin(k_j h/2), d_t -> -(2i/dt) sin(W dt/2),
                    for which a lattice plane wave is exactly null."""
    k = 2 * np.pi * np.fft.fftfreq(NG, d=H)
    k1 = (2 / H) * np.sin(k * H / 2) if native else k
    wt = (2 / DT) * np.sin(W * DT / 2) if native else W
    K = np.meshgrid(k1, k1, k1, indexing='ij')
    F = {c: np.fft.fftn(lock[CI[c]]) for c in COMP}
    hb = lambda m, n: F[(min(m, n), max(m, n))]
    tr = hb(1, 1) + hb(2, 2) + hb(3, 3)
    h00 = (hb(0, 0) + tr) / 2
    out = {}
    for model in ('GR', 'pub'):
        E = {}
        for i in range(3):
            for j in range(i, 3):
                hij = (hb(i + 1, j + 1) + (i == j) * (hb(0, 0) - tr) / 2) if model == 'GR' else (i == j) * h00
                e = -0.5 * (-wt ** 2 * hij - K[i] * K[j] * h00
                            + 1j * wt * (1j * K[j] * hb(0, i + 1) + 1j * K[i] * hb(0, j + 1)))
                E[(i, j)] = np.fft.ifftn(e).astype(np.complex64)
        out[model] = E
    harm = []
    for nu in range(4):
        dv = 1j * wt * hb(0, nu) + sum(1j * K[j] * hb(j + 1, nu) for j in range(3))
        harm.append(np.fft.ifftn(dv).astype(np.complex64))
    out['harm'] = harm
    out['scale'] = np.fft.ifftn(1j * wt * hb(0, 0)).astype(np.complex64)
    return out


def analyze_spectral(lock, T, XYZ, out):
    print("\n   re-analysis: spectral derivatives on grid points (continuum symbols | lattice-native symbols)")
    res = {}
    for native in (False, True):
        res[native] = spectral_fields(lock, native)
    rows = []
    for r in (20.0, 30.0, 40.0):
        for th in (0.0, 30.0, 60.0, 90.0):
            for phd in (0.0, 20.0, 45.0):
                if th == 0 and phd > 0:
                    continue
                t, p = np.radians(th), np.radians(phd)
                n0 = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
                g = np.round(r * n0 / H).astype(int)
                pt = g * H
                n = pt / np.linalg.norm(pt)
                gi = tuple(g + NG // 2)
                line = [np.linalg.norm(pt), th, phd]
                for native in (False, True):
                    S = res[native]
                    dec = {}
                    for model in ('GR', 'pub'):
                        E = np.zeros((3, 3), complex)
                        for (i, j), F in S[model].items():
                            E[i, j] = E[j, i] = F[gi]
                        dec[model] = decompose(E, n)
                    tt = np.hypot(abs(dec['GR']['plus']), abs(dec['GR']['cross']))
                    harm = max(abs(S['harm'][nu][gi]) for nu in range(4)) / max(abs(S['scale'][gi]), 1e-30)
                    line += [harm, tt, abs(dec['GR']['breath']) / tt, abs(dec['GR']['long']) / tt,
                             dec['GR']['vec'] / tt, np.hypot(abs(dec['pub']['plus']), abs(dec['pub']['cross'])) / tt,
                             abs(dec['pub']['breath']) / tt, abs(dec['pub']['long']) / tt, dec['pub']['vec'] / tt]
                rows.append(line)
    R = np.array(rows)
    out['spectral_rows'] = R
    print("   r     th  ph |  harm    GR breath  long    vec    | pub TT/TT breath long  vec  ||  native: harm   GR breath  long    vec")
    for x in R:
        print(f"   {x[0]:5.1f} {x[1]:3.0f} {x[2]:3.0f} | {x[3]:.1e} {x[5]:.1e} {x[6]:.1e} {x[7]:.1e} | "
              f"{x[8]:.3f} {x[9]:.3f} {x[10]:.3f} {x[11]:.3f} || {x[12]:.1e} {x[14]:.1e} {x[15]:.1e} {x[16]:.1e}")


def reanalyze(path):
    z = np.load(path)
    lock = [z[f'arr_{i}'] for i in range(len(COMP))]
    T, XYZ, err = source()
    out = dict(np.load(data_path('phase_bit_s4_open19.npz')))
    analyze_spectral(lock, T, XYZ, out)
    np.savez(data_path('phase_bit_s4_open19.npz'), **out)


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == 'reanalyze':
        reanalyze(sys.argv[2])
    else:
        main()
