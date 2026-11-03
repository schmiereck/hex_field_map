#!/usr/bin/env python3
"""
Paper 3, fourth follow-up to Experiment S4 — local full lapse, spatial metric
(Exp. 1c), origin of the vacuum counterterm, chiral branch of the interior node
==============================================================================

A.  Local full lapse by a wait buffer ("limited compute time per node"):
    every directed-edge state (x, d, s) gets a buffer partner.  One tick is
        U = R(a(x)) . [ S C  (+)  (-1) ]
    S C = ordinary walker step on the active register, the buffer only picks up
    the phase -1, and R(a) = cos a + i sin a sigma_x mixes active and buffer on
    the same site.  Strictly local and exactly unitary.  For every eigenphase
    theta of S C the tick has the eigenphase
        f_a(theta) = theta/2 + arcsin(cos a . sin(theta/2)),
    so near theta = 0 the whole quasi-energy is scaled by N = (1 + cos a)/2:
    a full lapse (coin AND propagation).  The buffer weight is 1 - N.
    Band level: formula vs exact spectrum, Kramers, buffer phase scan
    (only phi = pi keeps E = 0 fixed), counterterm offset scan for R'.
B.  x-chain (as in round 3): transverse acceleration with the buffer lapse
    N(x) = N0 (1 + g x), and Exp. 1c: buffer lapse N/A plus mass mu A
    (A = 1 - g x) -> H = (N/A) K + N M, the isotropic weak-field metric.
C.  E || B pumping in time: lowest Landau level at both Weyl nodes of one
    helicity sector (minimal SU(2), mu = 0), k_z swept linearly in time.

Run parts: python phase_bit_s4_open4.py [A] [B] [C]   (default: all)
"""

import functools
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_fcc_3d import MOVES_PHYS, DT_HALF, FCC_STEPS, LAT_A
from phase_bit_mass_s4 import PI, model_generators, candidates, bands_C
from phase_bit_s4_open import CP, alpha_eps
from phase_bit_s4_open3 import x_chain
from paths import data_path

print = functools.partial(print, flush=True)
EPS = 0.3
N0 = 0.75                                   # base lapse of the buffer (a0 = pi/3)


def f_dwell(theta, a):
    """Eigenphase of the buffered tick for the walker eigenphase theta."""
    return theta / 2 + np.arcsin(np.cos(a) * np.sin(theta / 2))


def a_of(N):
    return np.arccos(2 * np.asarray(N) - 1)


def dwell_k(k, C, a, phi=np.pi):
    """48x48 buffered tick at Bloch vector k (uniform a)."""
    u = np.repeat(np.exp(-1j * (MOVES_PHYS @ np.asarray(k, float))), 2)[:, None] * C
    I, Z = np.eye(24), np.zeros((24, 24))
    D = np.block([[u, Z], [Z, np.exp(-1j * phi) * I]])
    R = np.block([[np.cos(a) * I, 1j * np.sin(a) * I], [1j * np.sin(a) * I, np.cos(a) * I]])
    return R @ D


def model_2Ostar():
    models, _ = model_generators()
    W90 = candidates()['90° transport, K+K†']
    G2 = models['2O register']
    lvf = lambda l: 2 * np.sqrt(2) + l * np.sqrt(2)
    lam = brentq(lambda l: alpha_eps(G2 + l * W90, EPS, lvf(l)), -6, -2.2, xtol=1e-11)
    Gm, lv = G2 + lam * W90, lvf(lam)

    def X(mu, offset=0.0):
        """Coin generator with vacuum counterterm; offset = residual Dirac-point energy."""
        return EPS * Gm + mu * PI - (EPS * lv + offset * DT_HALF) * np.eye(24)
    return X, lam


def quartet(C, k=(0, 0, 0), E0=0.0):
    E = bands_C(np.asarray(k, float), C)[0]
    return np.sort(E[np.argsort(np.abs(E - E0))[:4]])


# ═════════════════════════════════════════════════════════════════════════════
def part_A(out):
    print("A. buffer lapse on the band level (2O*, eps = 0.3)")
    X, lam = model_2Ostar()
    print(f"   lambda*(eps) = {lam:.4f}")
    C = expm(1j * X(0.03))
    err, kram, unit = 0.0, 0.0, 0.0
    rng = np.random.default_rng(3)
    for _ in range(20):
        k = rng.uniform(-1.5, 1.5, 3)
        for a in (0.3, np.pi / 3, 1.2):
            Ud = dwell_k(k, C, a)
            th0 = -np.angle(np.linalg.eigvals(dwell_k(k, C, 0.0)[:24, :24]))
            lam_ = np.linalg.eigvals(Ud)
            err = max(err, max(np.min(np.abs(lam_ - np.exp(-1j * f_dwell(t, a)))) for t in th0))
            th = np.sort(-np.angle(lam_))
            kram = max(kram, np.abs(th[0::2] - th[1::2]).max())
            unit = max(unit, np.abs(Ud @ Ud.conj().T - np.eye(48)).max())
    print(f"   eigenphase formula f_a(theta) vs exact 48x48: max error {err:.1e}; unitarity {unit:.1e}; "
          f"Kramers pairs split by {kram:.1e} (20 random k, 3 values of a)")
    out['A_formula_err'], out['A_kramers'] = err, kram

    # quartet scaling and buffer weight
    q0 = quartet(C)
    a0 = a_of(N0)
    E_, V_ = np.linalg.eig(dwell_k(np.zeros(3), C, a0))
    Ed = -np.angle(E_) / DT_HALF
    idx = np.argsort(np.abs(Ed))[:4]
    wbuf = (np.abs(V_[24:, idx]) ** 2).sum(0) / (np.abs(V_[:, idx]) ** 2).sum(0)
    print(f"   N0 = {N0}: quartet {q0.round(4)} -> {np.sort(Ed[idx]).round(4)} (ratio {np.sort(Ed[idx])[3]/q0[3]:.4f}); "
          f"buffer weight {wbuf.mean():.4f} (1 - N0 = {1-N0})")
    out['A_bufweight'] = wbuf.mean()

    # buffer phase scan: does the vacuum stay at E = 0?
    C0 = expm(1j * X(0.0))
    phis = np.array([1.0, 0.95, 0.9, 0.75, 0.5])
    as_ = np.linspace(0.05, np.pi / 2, 30)
    vac = np.zeros((len(phis), len(as_)))
    for i, ph in enumerate(phis):
        for j, a in enumerate(as_):
            Ed = -np.angle(np.linalg.eigvals(dwell_k(np.zeros(3), C0, a, ph * np.pi))) / DT_HALF
            # Dirac quartet: continuation of theta = 0 on the active branch
            th = ph * np.pi
            e_pred = (-th / 2 + np.arccos(np.cos(a) * np.cos(th / 2))) / DT_HALF
            vac[i, j] = Ed[np.argmin(np.abs(Ed - e_pred))]
    out['A_phis'], out['A_as'], out['A_vac'] = phis, as_, vac
    print("   vacuum (Dirac point) energy after buffering, max over a in (0, pi/2]: " +
          ", ".join(f"phi = {p:.2f} pi: {np.abs(v).max():.2e}" for p, v in zip(phis, vac)))

    # counterterm offset scan: R' for particle and antiparticle, full lapse via the buffer
    print("   residual offset delta of the Dirac point (counterterm mistuned): R' particle / antiparticle")
    rows = []
    mu = 0.03
    for delta in (-0.06, -0.03, -0.01, 0.0, 0.01, 0.03, 0.06):
        Cd = expm(1j * X(mu, delta))
        h, dN = 1e-3, 1e-4

        def Ebuf(kx, N):
            th = quartet(Cd, (kx, 0, 0), delta) * DT_HALF
            return f_dwell(th, a_of(N)) / DT_HALF
        e0 = Ebuf(0, N0)
        dEdN = (Ebuf(0, N0 + dN) - Ebuf(0, N0 - dN)) / (2 * dN)
        curv = 2 * (Ebuf(h, N0) - e0) / h ** 2
        R = dEdN * curv / (N0 * CP ** 2)          # = E E'' / c'^2 for a pure N-scaling
        m = (e0[3] - e0[0]) / 2 / N0
        rows.append((delta, R[3], R[0], 1 + delta / m, 1 - delta / m))
        print(f"      delta = {delta:+.2f}: R' = {R[3]:+.3f} / {R[0]:+.3f}   (1 +- delta/m = {1+delta/m:+.3f} / {1-delta/m:+.3f})")
    out['A_offset'] = np.array(rows)


# ═════════════════════════════════════════════════════════════════════════════
def part_B(out):
    print("B. x-chain: buffer lapse (local full lapse) and Exp. 1c (spatial metric)")
    t0 = time.time()
    X, lam = model_2Ostar()
    Nx, g, SIG, nst = 160, 2e-3, 8.0, 120
    xs = (np.arange(Nx) - Nx // 2) * LAT_A
    ts = np.arange(nst + 1) * DT_HALF
    com = lambda m, b: (((np.abs(m) ** 2 + np.abs(b) ** 2).reshape(Nx, 24).sum(1)) @ xs) / \
        ((np.abs(m) ** 2).sum() + (np.abs(b) ** 2).sum())
    accel = lambda t, x: 2 * np.polyfit(t, x, 2)[0]

    def packet(Xm, kyz, sel):
        C = expm(1j * Xm)
        E, U = bands_C(np.array([0.0, *kyz]), C)
        uref = U[:, sel(E)[0]]
        uref = uref / np.linalg.norm(uref)
        psi = np.zeros((Nx, 24), complex)
        for kx in 2 * np.pi * np.fft.fftfreq(Nx) / LAT_A:
            gk = np.exp(-SIG ** 2 * kx ** 2 / 2)
            if gk < 1e-8:
                continue
            Ek, Uk = bands_C(np.array([kx, *kyz]), C)
            Q, _ = np.linalg.qr(Uk[:, sel(Ek)])
            u = Q @ (Q.conj().T @ uref)
            psi += gk * np.exp(1j * kx * xs)[:, None] * (u / np.linalg.norm(u))[None, :]
        return (psi / np.linalg.norm(psi)).ravel(), E[sel(E)].mean()

    cases = [('massive, rest', 0.15, (0.0, 0.0)), ('massive, k_y=0.3', 0.15, (0.3, 0.0)),
             ('massive, k_y=0.6', 0.15, (0.6, 0.0)), ('massless, k_y=0.2', 0.0, (0.2, 0.0)),
             ('massless, k_y=0.4', 0.0, (0.4, 0.0))]
    rows = []
    for cname, mu, kyz in cases:
        if mu > 0:
            sel = lambda E: np.sort(np.argsort(np.abs(E))[:4])[2:]
        else:
            kk = np.hypot(*kyz)
            sel = lambda E, kk=kk: np.sort(np.argsort(np.abs(E - CP * kk))[:2])
        psi_m0, Eband = packet(X(mu), kyz, sel)
        th_c = Eband * DT_HALF
        # group velocity along y of the band (for the GR prediction 1 + v^2/c'^2)
        Eky = lambda ky: (lambda E_: E_[sel(E_)].mean())(bands_C(np.array([0.0, ky, kyz[1]]), expm(1j * X(mu)))[0])
        vy = (Eky(kyz[0] + 1e-4) - Eky(kyz[0] - 1e-4)) / 2e-4
        res = [cname, vy / CP]
        for lname in ('buffer lapse', '1c metric'):
            if lname == 'buffer lapse':
                Nd = N0 * (1 + g * xs)
                mus = np.full(Nx, mu)
            else:
                Nd = N0 * (1 + g * xs) / (1 - g * xs)
                mus = mu * (1 - g * xs)
            a = a_of(Nd)
            Cs = [expm(1j * X(m_)) for m_ in mus] if mu > 0 else [expm(1j * X(0.0))] * Nx
            T = sp.csr_matrix(x_chain(Cs, kyz, Nx))
            ca, sa = np.repeat(np.cos(a), 24), np.repeat(np.sin(a), 24)

            def tick(m, b):
                m2, b2 = T @ m, -b
                return ca * m2 + 1j * sa * b2, 1j * sa * m2 + ca * b2
            # initial state on the physical branch: buffer amplitude from the local 2x2 eigenvector
            lam_c = np.exp(-1j * f_dwell(th_c, a))
            r = (np.cos(a) * np.exp(-1j * th_c) - lam_c) / (1j * np.sin(a))
            m, b = psi_m0.copy(), np.repeat(r, 24) * psi_m0
            ph = np.exp(1j * f_dwell(th_c, a_of(N0)))
            for _ in range(3):                        # spectral filter against the pi branch
                m2, b2 = tick(m, b)
                m, b = (m + ph * m2) / 2, (b + ph * b2) / 2
            nrm = np.sqrt(np.vdot(m, m).real + np.vdot(b, b).real)
            m, b = m / nrm, b / nrm
            xc = [com(m, b)]
            for _ in range(nst):
                m, b = tick(m, b)
                xc.append(com(m, b))
            asim = accel(ts, np.array(xc))

            # semiclassics with the exact band: H(x, k) = f_{a(x)}(theta(k; mu(x))) / dt
            def Hsc(kx, x):
                if lname == 'buffer lapse':
                    N_, mu_ = N0 * (1 + g * x), mu
                else:
                    N_, mu_ = N0 * (1 + g * x) / (1 - g * x), mu * (1 - g * x)
                E_ = bands_C(np.array([kx, *kyz]), expm(1j * X(mu_)))[0]
                return f_dwell(E_[sel(E_)].mean() * DT_HALF, a_of(N_)) / DT_HALF
            kq = np.linspace(-3 / SIG, 3 / SIG, 41)
            wq = np.exp(-SIG ** 2 * kq ** 2)
            hx, hk = 1.0, 1e-3
            Fx = np.array([(Hsc(k, hx) - Hsc(k, -hx)) / (2 * hx) for k in kq])
            d2 = np.array([(Hsc(k + hk, 0) - 2 * Hsc(k, 0) + Hsc(k - hk, 0)) / hk ** 2 for k in kq])
            asc = -np.sum(wq * Fx * d2) / wq.sum()
            res += [asim, asc]
            print(f"   {cname:18s} {lname:12s}: a_x = {asim:+.3e} (semicl. {asc:+.3e}); "
                  f"/(-g N0^2 c'^2) = {asim/(-g*N0**2*CP**2):.3f}  [{time.time()-t0:.0f}s]")
        res += [res[4] / res[2], res[5] / res[3], 1 + (vy / CP) ** 2]
        print(f"      ratio 1c / buffer: sim {res[-3]:.3f}, semicl. {res[-2]:.3f}, GR 1 + v^2/c'^2 = {res[-1]:.3f}  (v_y/c' = {vy/CP:.3f})")
        rows.append(tuple(res))
    out['B_rows'] = np.array(rows, dtype=object)
    out['B_g'], out['B_N0'] = g, N0

    # without counterterm: what the buffer lapse does to a rest particle (semiclassics)
    print("   without counterterm (Dirac point at E_c = +1.02): buffer lapse on a rest particle")
    Craw = expm(1j * (X(0.15) + EPS * (2 * np.sqrt(2) + lam * np.sqrt(2)) * np.eye(24)))
    Ec = -EPS * (2 * np.sqrt(2) + lam * np.sqrt(2)) / DT_HALF
    qr = quartet(Craw, E0=Ec)
    dN = 1e-4
    F = (f_dwell(qr * DT_HALF, a_of(N0 + dN)) - f_dwell(qr * DT_HALF, a_of(N0 - dN))) / (2 * dN) / DT_HALF
    Fct = (f_dwell((qr - Ec) * DT_HALF, a_of(N0 + dN)) - f_dwell((qr - Ec) * DT_HALF, a_of(N0 - dN))) / (2 * dN) / DT_HALF
    print(f"      dE'/dN upper doublet: {F[3]:+.3f} without vs {Fct[3]:+.3f} with counterterm "
          f"(ratio {F[3]/Fct[3]:.1f}); lower doublet {F[0]:+.3f} vs {Fct[0]:+.3f}")
    out['B_noct'] = np.array([F[3], Fct[3], F[0], Fct[0]])


# ═════════════════════════════════════════════════════════════════════════════
def part_C(out):
    print("C. E || B pumping in time: lowest Landau level at both nodes (helicity +1, minimal SU(2), mu = 0)")
    t0 = time.time()
    models, _ = model_generators()
    H = candidates()['on-site u_d.s']
    G = models['minimal SU(2)']
    C = expm(1j * EPS * G)
    wH, VH = np.linalg.eigh(H)
    nodes = (-4 * np.sqrt(3) * EPS, 4 * EPS / np.sqrt(3))
    for B, Nx in ((0.05, 200), (0.035, 260)):
        xs = (np.arange(Nx) - Nx // 2) * LAT_A
        src = []
        for d in range(12):
            l = np.arange(Nx)
            l2 = l + FCC_STEPS[d, 0]
            ok = (l2 >= 0) & (l2 < Nx)
            src.append((d, l[ok], l2[ok], xs[l[ok]] + 0.5 * MOVES_PHYS[d, 0]))

        def step(psi, kz):
            phi = (psi @ C.T).reshape(Nx, 12, 2)
            o = np.zeros_like(phi)
            for d, l, l2, xm in src:
                o[l2, d] += np.exp(-1j * (B * xm * MOVES_PHYS[d, 1] + kz * MOVES_PHYS[d, 2]))[:, None] * phi[l, d]
            return o.reshape(Nx, 24)
        lB = 1 / np.sqrt(B)
        mz = np.tile(np.repeat(MOVES_PHYS[:, 2], 2), Nx)
        Ps = np.kron(np.eye(Nx), VH[:, np.isclose(wH, 1)])
        Ts = Ps.conj().T @ x_chain([C] * Nx, (0.0, 0.0), Nx, B=B, xs=xs, periodic=False) @ Ps
        lamv, U = np.linalg.eig(Ts)
        del Ts
        E = -np.angle(lamv) / DT_HALF
        wl = (np.abs(U) ** 2).reshape(Nx, 12, -1).sum(1)
        cen = wl[np.abs(xs) < 3 * lB].sum(0) / wl.sum(0)
        good = np.where((np.abs(lamv) > 0.999) & (cen > 0.6))[0]
        for En in nodes:
            cand = good[np.abs(E[good] - En) < 0.08]
            V = Ps @ U[:, cand]
            V /= np.linalg.norm(V, axis=0)
            vz = (mz[:, None] * np.abs(V) ** 2).sum(0) / DT_HALF
            j = np.argmax(np.abs(vz))
            psi0 = V[:, j].reshape(Nx, 24)
            tr = {}
            for sgn in (+1, -1):
                Fz = 2e-4
                psi, kz, rec = psi0.copy(), 0.0, []
                for it in range(int(0.25 / Fz) + 1):
                    if it % 50 == 0:
                        z = np.vdot(psi, step(psi, kz))
                        nrm = np.vdot(psi, psi).real
                        rec.append((kz, -np.angle(z) / DT_HALF - En, abs(z) / nrm))
                    psi = step(psi, kz)
                    kz += sgn * Fz
                tr[sgn] = np.array(rec)
            both = np.vstack([tr[-1][::-1], tr[+1][1:]])
            sl = np.polyfit(both[np.abs(both[:, 0]) <= 0.05, 0], both[np.abs(both[:, 0]) <= 0.05, 1], 1)[0]
            out[f'C_B{B}_E{En:+.3f}'] = both
            out[f'C_B{B}_E{En:+.3f}_v'] = np.array([vz[j], sl, E[cand[j]] - En])
            print(f"   B = {B}, node E = {En:+.3f}: LLL at E - E_node = {E[cand[j]]-En:+.4f}; v_z from the state "
                  f"{vz[j]/CP:+.3f} c'; pumped dE/dk_z = {sl/CP:+.3f} c'; min |<U>| along the sweep "
                  f"{both[:, 2].min():.4f}  [{time.time()-t0:.0f}s]")


def main():
    parts = [p.upper() for p in sys.argv[1:]] or ['A', 'B', 'C']
    path = data_path('phase_bit_s4_open4.npz')
    out = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}
    for p in parts:
        {'A': part_A, 'B': part_B, 'C': part_C}[p](out)
        np.savez(path, **out)
    print("saved data/phase_bit_s4_open4.npz")


if __name__ == '__main__':
    main()
