#!/usr/bin/env python3
"""
Paper 3, H2 continued: compensating the stay-gate background, and beta_PPN
==============================================================================

A.  Compensation.  The stay gate of H2 adds a background -alpha k^2 with
    alpha = 0.245 N_h (1 - N_h).  The W90 coupling lambda (round 2) changes only
    the background (gap, Dirac point, velocity untouched), so a loaded node
    retunes lambda(N_h) such that alpha(mu = 0) is back at its vacuum value.
    A1  lambda(N_h) by root finding; checks for mu = 0, 0.03, 0.15.
B.  x-chain deflection (as open8 part B) with the compensated coin per layer:
    H2c vs wait only (1b), against the ideal metric with gamma = 1.
C.  beta_PPN.  ln N = -U + (beta - 1) U^2 for g00 = -N^2 = -1 + 2U - 2 beta U^2,
    so for two sources  beta - 1 = [ln N12 - ln N1 - ln N2] / (2 ln N1 ln N2).
    C1  composition of two loads at one node (band level, clock rate = gap/m):
        one gate with linearly superposed Phi (beta 1/2), parallel wait
        registers (waits add, beta 3/2), nested gates (product, beta 1);
        nested stay gates for the hop (A = 1/(N1 N2)).
    C2  source redshift: rest load Q = E and mass load Q_M = mu dE/dmu of the
        H2c scheme at uniform N (the active mass of a source sitting in a
        potential) -> the Phi_2 term of the PPN metric.

Run parts: python phase_bit_s4_open9.py [A] [B] [C]   (default: all)
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
from phase_bit_mass_s4 import PI, model_generators, candidates
from phase_bit_s4_open import CP
from phase_bit_s4_open4 import model_2Ostar, a_of, N0, EPS
from phase_bit_s4_open5 import chain_T
from phase_bit_s4_open8 import Scheme, bloch_U, energies, near, ONE
from paths import data_path

print = functools.partial(print, flush=True)
_models, _ = model_generators()
W90 = candidates()['90° transport, K+K†']
G2 = _models['2O register']
LAM0 = model_2Ostar()[1]
lvf = lambda l: 2 * np.sqrt(2) + l * np.sqrt(2)


def Xl(lam):
    """2O* coin generator at W90 coupling lam, counterterm keeps the Dirac point at 0."""
    return lambda mu, offset=0.0: EPS * (G2 + lam * W90) + mu * PI - EPS * lvf(lam) * np.eye(24)


def band_alpha(lam, Nh, mu, k=0.01):
    X = Xl(lam)
    bd = lambda q: near(energies(bloch_U(X, 'WH', (0, q, 0), mu, ONE, Nh)), np.hypot(Nh * CP * q, 2 * mu), 2).mean()
    ED = lambda q: np.hypot(Nh * CP * q, 2 * mu)
    return -((bd(k) - bd(0)) - (ED(k) - ED(0))) / k ** 2


def shift_chain(kyz, Nx):
    """sparse x-chain shift alone (open ends): layer l -> l + dl for heading d."""
    P = None
    for dl in (-1, 0, 1):
        dg = np.zeros(24, complex)
        for d in np.where(FCC_STEPS[:, 0] == dl)[0]:
            dg[2 * d:2 * d + 2] = np.exp(-1j * (kyz[0] * MOVES_PHYS[d, 1] + kyz[1] * MOVES_PHYS[d, 2]))
        blk = sp.kron(sp.eye(Nx, k=-dl, format='csr'), sp.diags(dg), format='csr')
        P = blk if P is None else P + blk
    return P


# ═════════════════════════════════════════════════════════════════════════════
def part_A(out):
    print(f"A. compensating the stay-gate background by lambda(N_h)  (lambda* = {LAM0:.4f})")
    t0 = time.time()
    target = band_alpha(LAM0, ONE, 0.0)
    own03, own15 = band_alpha(LAM0, ONE, 0.03), band_alpha(LAM0, ONE, 0.15)
    print(f"   vacuum: alpha(mu = 0) = {target:+.5f}, alpha(0.03) = {own03:+.5f}, alpha(0.15) = {own15:+.5f}")
    rows = []
    for Nh in (0.99, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.6, 0.5, 0.45, 0.4):
        lam = brentq(lambda l: band_alpha(l, Nh, 0.0) - target, LAM0, -2.02, xtol=1e-8)
        X = Xl(lam)
        vac = np.abs(energies(bloch_U(X, 'WH', (0, 0, 0), 0.0, ONE, Nh))).min()
        gap = near(energies(bloch_U(X, 'WH', (0, 0, 0), 0.03, ONE, Nh)), 0.0, 4)[-1] / 0.06
        vs = [near(energies(bloch_U(X, 'WH', (0, q, 0), 0.0, ONE, Nh)), 0.0, 4)[-1] / (CP * q * Nh) for q in (0.01, 0.02)]
        v = 2 * vs[0] - vs[1]
        a_before = [band_alpha(LAM0, Nh, mu) for mu in (0.03, 0.15)]
        a_after = [band_alpha(lam, Nh, mu) for mu in (0.03, 0.15)]
        rows.append((Nh, lam, vac, gap, v, *a_before, *a_after))
        print(f"   N_h = {Nh:.2f}: lambda = {lam:+.5f} (d = {lam-LAM0:+.4f}); Dirac point {vac:.1e}, gap/m {gap:.5f}, "
              f"v/(N_h c') {v:.5f}; alpha(0.03) {a_before[0]:+.4f} -> {a_after[0]:+.4f} (vac {own03:+.4f}), "
              f"alpha(0.15) {a_before[1]:+.4f} -> {a_after[1]:+.4f} (vac {own15:+.4f})  [{time.time()-t0:.0f}s]")
    r = np.array(rows)
    out['A1'] = r
    out['A_own'] = np.array([target, own03, own15])
    # smooth representation d lambda = sum_n c_n (1 - N)^n for the chain
    c = np.polyfit(1 - r[:, 0], r[:, 1] - LAM0, 4)
    out['A_fit'] = c
    res = np.abs(np.polyval(c, 1 - r[:, 0]) - (r[:, 1] - LAM0)).max()
    print(f"   fit d lambda(1 - N) (degree 4, through the data): max residual {res:.1e}; weak field "
          f"d lambda = {np.polyval(np.polyder(c), 0):.3f} (1 - N)")
    try:
        brentq(lambda l: band_alpha(l, 0.3, 0.0) - target, LAM0, -2.02)
    except ValueError:
        print("   N_h = 0.3: no root before lambda = -2 (the two quartets meet) -> compensation limited to N_h >~ 0.4")


def lam_of(Nh, c):
    return LAM0 + np.polyval(c, 1 - np.asarray(Nh))


# ═════════════════════════════════════════════════════════════════════════════
def part_B(out):
    print("B. x-chain: wait only (1b) vs compensated H2 (H2c), N(x) = N0 (1 + g x)")
    t0 = time.time()
    c = out['A_fit']
    Nx, g, SIG, nst = 300, 2e-3, 16.0, 160
    xs = (np.arange(Nx) - Nx // 2) * LAT_A
    ts = np.arange(nst + 1) * DT_HALF
    accel = lambda t, x: 2 * np.polyfit(t, x, 2)[0]
    Nx_ = N0 * (1 + g * xs)
    # check the sparse construction against chain_T for a uniform coin
    Ct = expm(1j * Xl(LAM0)(0.03))
    P = shift_chain((0.2, 0.0), 40)
    dT = abs(P @ sp.kron(sp.eye(40), sp.csr_matrix(Ct)) - chain_T(Ct, (0.2, 0.0), 40)).max()
    print(f"   shift x block-diagonal coin vs chain_T: {dT:.1e}")
    cases = [('mu=0.03, rest', 0.03, 0.0), ('mu=0.03, k_y=0.1', 0.03, 0.1), ('mu=0.03, k_y=0.2', 0.03, 0.2),
             ('mu=0.15, rest', 0.15, 0.0), ('mu=0.15, k_y=0.3', 0.15, 0.3),
             ('massless, k_y=0.1', 0.0, 0.1), ('massless, k_y=0.2', 0.0, 0.2), ('massless, k_y=0.4', 0.0, 0.4)]
    # scheme: kind, (N_w, N_h) from N, lambda from N
    schemes = {'1b': ('W', lambda N: (N, ONE * np.ones_like(N)), lambda N: LAM0 * np.ones_like(N)),
               'H2c': ('WH', lambda N: (N, N), lambda N: lam_of(N, c))}

    def band(kind, Nf, lf, mu, k, N):
        Nw, Nh = Nf(np.float64(N))
        E = energies(bloch_U(Xl(float(lf(np.float64(N)))), kind, k, mu, float(Nw), float(Nh)))
        kk = np.linalg.norm(k)
        guess = Nw * np.hypot(Nh * CP * kk, 2 * mu) if mu > 0 else Nw * Nh * CP * kk
        return near(E, guess, 2).mean()

    kq = np.linspace(-3 / SIG, 3 / SIG, 25)
    wq = np.exp(-SIG ** 2 * kq ** 2)
    rows, traj = [], {}
    for cname, mu, ky in cases:
        res = [cname]
        for sname, (kind, Nf, lf) in schemes.items():
            nr = Scheme(kind, None, None).nreg()
            Nw0, Nh0 = Nf(np.float64(N0))
            X0 = Xl(float(lf(np.float64(N0))))
            Eb = band(kind, Nf, lf, mu, (0, ky, 0), N0)
            E0, V0 = np.linalg.eig(bloch_U(X0, kind, (0, ky, 0), mu, float(Nw0), float(Nh0)))
            uref = V0[:, np.argsort(np.abs(-np.angle(E0) / DT_HALF - Eb))[0]]
            psi = np.zeros((Nx, 24 * nr), complex)
            for kx in 2 * np.pi * np.fft.fftfreq(Nx) / LAT_A:
                gk = np.exp(-SIG ** 2 * kx ** 2 / 2)
                if gk < 1e-8:
                    continue
                Ek, Vk = np.linalg.eig(bloch_U(X0, kind, (kx, ky, 0), mu, float(Nw0), float(Nh0)))
                Ek = -np.angle(Ek) / DT_HALF
                Q, _ = np.linalg.qr(Vk[:, np.argsort(np.abs(Ek - band(kind, Nf, lf, mu, (kx, ky, 0), N0)))[:2]])
                u = Q @ (Q.conj().T @ uref)
                psi += gk * np.exp(1j * kx * xs)[:, None] * (u / np.linalg.norm(u))[None, :]
            psi /= np.linalg.norm(psi)
            lams = lf(Nx_)
            Cs = [expm(1j * Xl(l)(mu)) for l in lams]
            Cb = sp.block_diag([sp.csr_matrix(C_) for C_ in Cs], format='csr')
            sc = Scheme(kind, shift_chain((ky, 0.0), Nx) @ Cb, Cb)
            Nw, Nh = Nf(Nx_)
            sc.set_N(Nw, Nh)
            st = [psi[:, 24 * q:24 * (q + 1)].ravel() for q in range(nr)]
            ph = np.exp(1j * Eb * DT_HALF)
            for _ in range(3):
                st2 = sc.tick(st)
                st = [(a + ph * b) / 2 for a, b in zip(st, st2)]
            nrm = np.sqrt(sum(np.vdot(a, a).real for a in st))
            st = [a / nrm for a in st]

            def com(st):
                w = sum((np.abs(a) ** 2).reshape(Nx, 24).sum(1) for a in st)
                return (w @ xs) / w.sum()
            xc = [com(st)]
            for _ in range(nst):
                st = sc.tick(st)
                xc.append(com(st))
            xc = np.array(xc)
            asim = accel(ts, xc)
            traj[f'{cname}|{sname}'] = xc
            Hsc = lambda kx, x: band(kind, Nf, lf, mu, (kx, ky, 0), N0 * (1 + g * x))
            Fx = np.array([(Hsc(k, 1.0) - Hsc(k, -1.0)) / 2 for k in kq])
            d2 = np.array([(Hsc(k + 1e-3, 0) - 2 * Hsc(k, 0) + Hsc(k - 1e-3, 0)) / 1e-6 for k in kq])
            asc = -np.sum(wq * Fx * d2) / wq.sum()
            res += [asim, asc]
            print(f"   {cname:18s} {sname:3s}: a_x = {asim:+.3e} (semicl. {asc:+.3e})  [{time.time()-t0:.0f}s]")
        a1, s1, a2, s2 = res[1:]
        R, Rs = (a2 / N0 ** 2) / a1, (s2 / N0 ** 2) / s1

        def a_ideal(gam, base=True):
            def H(kx, x):
                Nl = N0 * (1 + g * x)
                A = (1 / N0) * (Nl / N0) ** (-gam) if base else 1.0
                return Nl * np.hypot(2 * mu, CP * np.hypot(kx, ky) / A)
            Fx = np.array([(H(k, 1.0) - H(k, -1.0)) / 2 for k in kq])
            d2 = np.array([(H(k + 1e-3, 0) - 2 * H(k, 0) + H(k - 1e-3, 0)) / 1e-6 for k in kq])
            return -np.sum(wq * Fx * d2) / wq.sum()
        gg = np.linspace(-1.0, 3.0, 81)
        Rid = np.array([(a_ideal(x) / N0 ** 2) / a_ideal(0.0, base=False) for x in gg])
        R_gr = np.interp(1.0, gg, Rid)
        sens = R_gr - np.interp(0.0, gg, Rid)
        o = np.argsort(Rid)
        fit = lambda r: np.interp(r, Rid[o], gg[o]) if (abs(sens) > 0.1 and Rid.min() <= r <= Rid.max()) else np.nan
        res += [R, Rs, R_gr, fit(R), fit(Rs)]
        print(f"      (a_H2c / N0^2) / a_1b: sim {R:.3f}, lattice semicl. {Rs:.3f}; ideal gamma = 1: {R_gr:.3f}"
              + (f"; fitted gamma: sim {fit(R):.3f}, band {fit(Rs):.3f}" if np.isfinite(fit(R)) else ""))
        rows.append(res)
    out['B_rows'] = np.array(rows, dtype=object)
    out['B_ts'] = ts

    # B2: residual at mu = 0.15 = walker's own (mass-dependent) background, not rescaled by N^2
    print("   B2  mu = 0.15: compensated alpha_c(N) vs the walker's own alpha (lambda tuned at mu = 0)")
    Ns = np.linspace(0.55, 0.95, 9)
    As = np.array([band_alpha(lam_of(N, c), N, 0.15) for N in Ns])
    own = band_alpha(LAM0, ONE, 0.15)
    print(f"      alpha_c(N = {Ns[0]:.2f} ... {Ns[-1]:.2f}) = {As[0]:.4f} ... {As[-1]:.4f}, vacuum {own:.4f}")
    b2 = []
    for ky in (0.0, 0.3):
        m = 0.3
        H1 = lambda kx, x: N0 * (1 + g * x) * (np.hypot(m, CP * np.hypot(kx, ky)) - own * (kx ** 2 + ky ** 2))
        HA = lambda kx, x, sc: (lambda N: N * (np.hypot(m, CP * np.hypot(kx, ky) * N)
                                          - (own * N ** 2 if sc else np.interp(N, Ns, As)) * (kx ** 2 + ky ** 2)))(N0 * (1 + g * x))
        acc = lambda H: -np.sum(wq * np.array([(H(k, 1.0) - H(k, -1.0)) / 2 for k in kq]) *
                                np.array([(H(k + 1e-3, 0) - 2 * H(k, 0) + H(k - 1e-3, 0)) / 1e-6 for k in kq])) / wq.sum()
        r1 = acc(lambda k, x: HA(k, x, False)) / N0 ** 2 / acc(H1)
        r2 = acc(lambda k, x: HA(k, x, True)) / N0 ** 2 / acc(H1)
        b2.append((ky, r1, r2))
        print(f"      k_y = {ky}: gamma = 1 + alpha_c(N): {r1:.3f};  gamma = 1 with the own background scaled by N^2: {r2:.3f}")
    out['B2'] = np.array(b2)
    for key, v in traj.items():
        out['B_traj_' + key] = v


# ═════════════════════════════════════════════════════════════════════════════
def U_compose(X, law, k, mu, N1, N2):
    """Bloch tick with two loads at one node.  law: 'linear' (one gate, Phi superposed linearly),
    'parallel' (two wait registers on the active amplitude), 'nested' (gate 2 also acts on register 1),
    'nested_hop' (H2: nested wait AND nested stay gates)."""
    C = expm(1j * X(mu))
    U0 = Sk(k) @ C
    I24 = np.eye(24)

    def R(a, n):
        c, s = np.cos(a), np.sin(a)
        In = np.eye(n)
        return np.block([[c * In, 1j * s * In], [1j * s * In, c * In]])
    if law == 'linear':
        N = 1 - (1 - N1) - (1 - N2)
        return R(a_of(N), 24) @ np.block([[U0, 0 * I24], [0 * I24, -I24]])
    if law == 'parallel':
        # registers (m, b1, b2): b1 mixes with m by a1, then b2 mixes with m by a2
        Z = np.zeros((24, 24))
        D = np.block([[U0, Z, Z], [Z, -I24, Z], [Z, Z, -I24]])
        c1, s1, c2, s2 = np.cos(a_of(N1)), np.sin(a_of(N1)), np.cos(a_of(N2)), np.sin(a_of(N2))
        R1 = np.block([[c1 * I24, 1j * s1 * I24, Z], [1j * s1 * I24, c1 * I24, Z], [Z, Z, I24]])
        R2 = np.block([[c2 * I24, Z, 1j * s2 * I24], [Z, I24, Z], [1j * s2 * I24, Z, c2 * I24]])
        return R2 @ R1 @ D
    if law == 'nested':
        inner = R(a_of(N1), 24) @ np.block([[U0, 0 * I24], [0 * I24, -I24]])
        return R(a_of(N2), 48) @ np.block([[inner, np.zeros((48, 48))], [np.zeros((48, 48)), -np.eye(48)]])
    if law == 'nested_hop':
        # stay gates for both loads (nested), then wait gates for both loads (nested)
        V = R(a_of(N1), 24) @ np.block([[U0, 0 * I24], [0 * I24, -C]])
        Cn = np.kron(np.eye(2), C)
        V = R(a_of(N2), 48) @ np.block([[V, np.zeros((48, 48))], [np.zeros((48, 48)), -Cn]])
        for Nn in (N1, N2):
            n = V.shape[0]
            V = R(a_of(Nn), n) @ np.block([[V, np.zeros((n, n))], [np.zeros((n, n)), -np.eye(n)]])
        return V


def Sk(k):
    return np.diag(np.repeat(np.exp(-1j * (MOVES_PHYS @ np.asarray(k, float))), 2))


def part_C(out):
    print("C. beta_PPN: two loads at one node; clock rate N = gap/m (mu = 0.03), hop from the massless slope")
    X = Xl(LAM0)
    m = 0.06
    gap = lambda law, N1, N2: near(energies(U_compose(X, law, (0, 0, 0), 0.03, N1, N2)), 0.0, 4)[-1] / m
    rows = []
    for N1, N2 in ((0.99, 0.99), (0.98, 0.95), (0.9, 0.9), (0.8, 0.9), (0.7, 0.7)):
        for law, pred in (('linear', 0.5), ('parallel', 1.5), ('nested', 1.0)):
            g1, g2, g12 = gap(law, N1, ONE), gap(law, ONE, N2), gap(law, N1, N2)
            L1, L2, L12 = np.log(g1), np.log(g2), np.log(g12)
            beta = 1 + (L12 - L1 - L2) / (2 * L1 * L2)
            rows.append((N1, N2, ['linear', 'parallel', 'nested'].index(law), g1, g2, g12, beta, pred))
            print(f"   N1 = {N1:.2f}, N2 = {N2:.2f}, {law:8s}: N(1) = {g1:.5f}, N(2) = {g2:.5f}, N(1+2) = {g12:.5f}; "
                  f"beta = {beta:.4f} (weak-field law: {pred})")
    out['C1'] = np.array(rows)

    # H2 with two loads: nested stay + wait gates -> clock N1 N2, hop (N1 N2)^2
    print("   H2 with two loads (nested stay and wait gates):")
    hrows = []
    for N1, N2 in ((0.95, 0.9), (0.8, 0.7)):
        g12 = near(energies(U_compose(X, 'nested_hop', (0, 0, 0), 0.03, N1, N2)), 0.0, 4)[-1] / m
        vs = [near(energies(U_compose(X, 'nested_hop', (0, q, 0), 0.0, N1, N2)), 0.0, 4)[-1] / (CP * q) for q in (0.01, 0.02)]
        v = 2 * vs[0] - vs[1]
        hrows.append((N1, N2, g12, v))
        print(f"      N1 = {N1}, N2 = {N2}: clock {g12:.5f} (N1 N2 = {N1*N2:.5f}), hop {v:.5f} ((N1 N2)^2 = {(N1*N2)**2:.5f}) "
              f"-> A = {g12/v:.4f} vs 1/(N1 N2) = {1/(N1*N2):.4f}")
    out['C1_hop'] = np.array(hrows)

    # C2: active mass of a source sitting at lapse N (H2c scheme, Hellmann-Feynman on the band)
    print("   C2: load of a rest source at uniform lapse N (H2 scheme): Q = E, Q_M = mu dE/dmu, source 2Q - Q_M")
    c = out['A_fit'] if 'A_fit' in out else None
    crow = []
    for N in (0.999, 0.9, 0.75, 0.5):
        lam = lam_of(N, c) if c is not None and N >= 0.4 else LAM0
        Xc = Xl(lam)
        E = lambda mu: near(energies(bloch_U(Xc, 'WH', (0, 0, 0), mu, N, N)), N * 2 * mu, 2).mean()
        mu, dmu = 0.03, 1e-4
        Q = E(mu)
        QM = mu * (E(mu + dmu) - E(mu - dmu)) / (2 * dmu)
        crow.append((N, Q / m, QM / m, (2 * Q - QM) / m))
        print(f"      N = {N:.3f}: Q/m = {Q/m:.5f}, Q_M/m = {QM/m:.5f}, source/m = {(2*Q-QM)/m:.5f}  (N = {N})")
    out['C2'] = np.array(crow)


def main():
    parts = [p.upper() for p in sys.argv[1:]] or ['A', 'B', 'C']
    path = data_path('phase_bit_s4_open9.npz')
    out = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}
    for p in parts:
        {'A': part_A, 'B': part_B, 'C': part_C}[p](out)
        np.savez(path, **out)
    print("saved data/phase_bit_s4_open9.npz")


if __name__ == '__main__':
    main()
