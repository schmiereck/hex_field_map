#!/usr/bin/env python3
"""
Paper 3, hypothesis H2: one budget rule per node for lapse AND spatial metric
==============================================================================

Rule: "every process needs each node it involves to be free".  A node is free
with weight N(x) (the wait-buffer lapse of round 4).  The internal clock (coin,
mass rotation) involves one node, a hop involves two (departure and arrival).

Two local gates, both phase-pi registers with mixing angle a = arccos(2N - 1):
  wait  W(a): R(a)[V (+) -1]      the node is busy, NOTHING is computed
  stay  H(b): R(b)[S C (+) -C]    the hop is refused, the clock (coin) still runs
H2 tick:  U = W(a(x)) [ H(b(x)) (S C (+) -C) (+) -1 ]   with N_w = N_h = N(x).

A.  Band level (Bloch matrices).
    A1  no-go: ANY combination of wait buffers (also separate buffers for the
        shift and the coin) scales hop and clock by the same factor
        1/(1/N1 + 1/N2 - 1) -> pure lapse, gamma = 0.
    A2  the stay gate scales only the hop: v = c' N_w N_h, gap = 2 mu N_w
        -> H = N^2 K + N M = Exp. 1c with A = 1/N, gamma_PPN = 1.
B.  x-chain, N(x) = N0 (1 + g x): transverse acceleration for rest, moving and
    massless packets, wait only (1b) vs H2; GR: (a_H2 / N0^2) / a_1b = 1 + gamma v^2.

Run parts: python phase_bit_s4_open8.py [A] [B]   (default: both)
"""

import functools
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.linalg import expm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quantum_fcc_3d import MOVES_PHYS, DT_HALF, LAT_A
from phase_bit_s4_open import CP
from phase_bit_s4_open4 import model_2Ostar, a_of, N0
from phase_bit_s4_open5 import chain_T
from paths import data_path

print = functools.partial(print, flush=True)
ONE = 1 - 1e-13


def Sk(k):
    return np.diag(np.repeat(np.exp(-1j * (MOVES_PHYS @ np.asarray(k, float))), 2))


class Scheme:
    """Register tick for 'W' (wait), 'H' (stay), 'WH' (both), 'WW' (separate wait
    buffers for shift and coin).  Works with dense Bloch blocks or sparse chains:
    T = shift . coin, Cb = coin, S = shift alone; gate angles per state."""

    def __init__(self, kind, T, Cb, S=None):
        self.kind, self.T, self.Cb, self.S = kind, T, Cb, S

    def set_N(self, Nw, Nh, rep=24):
        f = lambda N: (np.repeat(np.cos(a_of(N)), rep), np.repeat(np.sin(a_of(N)), rep))
        self.cw, self.sw = f(np.atleast_1d(Nw))
        self.ch, self.sh = f(np.atleast_1d(Nh))

    def nreg(self):
        return {'W': 2, 'H': 2, 'WH': 4, 'WW': 3}[self.kind]

    def tick(self, st):
        k = self.kind
        if k == 'WW':                          # (m, buffer of the coin, buffer of the shift)
            m, bc, bs = st
            mc = self.Cb @ m
            m1, bc1 = self.ch * mc + 1j * self.sh * (-bc), 1j * self.sh * mc + self.ch * (-bc)
            ms = self.S @ m1
            return [self.cw * ms + 1j * self.sw * (-bs), bc1, 1j * self.sw * ms + self.cw * (-bs)]
        m = st[0]
        m1 = self.T @ m
        if k in ('H', 'WH'):
            s1 = -(self.Cb @ st[1])
            inner = [self.ch * m1 + 1j * self.sh * s1, 1j * self.sh * m1 + self.ch * s1]
        else:
            inner = [m1]
        if k == 'H':
            return inner
        waits = st[len(inner):]
        return [self.cw * x + 1j * self.sw * (-w) for x, w in zip(inner, waits)] + \
               [1j * self.sw * x + self.cw * (-w) for x, w in zip(inner, waits)]


def bloch_U(X, kind, k, mu, Nw, Nh):
    C = expm(1j * X(mu))
    S = Sk(k)
    sc = Scheme(kind, S @ C, C, S)
    sc.set_N(Nw, Nh)
    n = 24 * sc.nreg()
    cols = []
    for j in range(n):
        e = np.zeros(n, complex)
        e[j] = 1
        out = sc.tick(list(e.reshape(sc.nreg(), 24)))
        cols.append(np.concatenate(out))
    return np.array(cols).T


def energies(U):
    return -np.angle(np.linalg.eigvals(U)) / DT_HALF


def near(E, guess, n=2):
    return np.sort(E[np.argsort(np.abs(E - guess))[:n]])


def alpha_bg(X, Nh, mu, k=0.01):
    """background -alpha k^2 of the H-gated band against Dirac with velocity N_h c' (N_w = 1)."""
    bd = lambda q: near(energies(bloch_U(X, 'WH', (0, q, 0), mu, ONE, Nh)), np.hypot(Nh * CP * q, 2 * mu), 2).mean()
    ED = lambda q: np.hypot(Nh * CP * q, 2 * mu)
    return -((bd(k) - bd(0)) - (ED(k) - ED(0))) / k ** 2


# ═════════════════════════════════════════════════════════════════════════════
def part_A(out):
    print("A. band level: wait vs stay (2O*, eps = 0.3); gap at k = 0 (mu = 0.03, m = 0.06), "
          "massless slope at |k| = 0.02")
    X, lam = model_2Ostar()
    m, kk = 0.06, 0.02
    rows = []
    for kind in ('W', 'WW', 'H', 'WH'):
        for Nw, Nh in ((0.75, ONE), (ONE, 0.75), (0.75, 0.75), (0.5, 0.6), (0.4, 0.4), (0.9, 0.9)):
            if kind == 'W' and Nh < 1 - 1e-9:
                continue
            if kind == 'H' and Nw < 1 - 1e-9:
                continue
            E0 = energies(bloch_U(X, kind, (0, 0, 0), 0.0, Nw, Nh))
            vac = np.abs(E0).min()
            gap = near(energies(bloch_U(X, kind, (0, 0, 0), 0.03, Nw, Nh)), 0.0, 4)[-1] / m
            v = near(energies(bloch_U(X, kind, (0, kk, 0), 0.0, Nw, Nh)), 0.0, 4)[-1] / (CP * kk)
            if kind in ('W', 'WW'):
                pg = pv = 1 / (1 / Nw + 1 / Nh - 1)
            else:
                pg, pv = Nw, Nw * Nh
            rows.append((['W', 'WW', 'H', 'WH'].index(kind), Nw, Nh, gap, v, pg, pv, vac))
            print(f"   {kind:3s} N_w = {Nw:.2f}, N_h = {Nh:.2f}: gap/m = {gap:.4f} (pred {pg:.4f}), "
                  f"v/c' = {v:.4f} (pred {pv:.4f}); Dirac point E = {vac:.1e}")
    out['A1'] = np.array(rows)

    # gamma from the band: H = (N/A) K + N M  ->  A = gap/v, N = gap; A = N^(-gamma)
    print("   H2 (N_w = N_h = N): effective spatial metric A = (gap/m)/(v/c'), gamma = -ln A / ln N")
    g_rows = []
    for N in (0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3):
        gap = near(energies(bloch_U(X, 'WH', (0, 0, 0), 0.03, N, N)), 0.0, 4)[-1] / m
        vs = []
        for kq in (0.01, 0.02):
            vs.append(near(energies(bloch_U(X, 'WH', (0, kq, 0), 0.0, N, N)), 0.0, 4)[-1] / (CP * kq))
        v = 2 * vs[0] - vs[1]                     # remove the O(k^2) part of the slope
        A = gap / v
        gam = -np.log(A) / np.log(N)
        # weak field: N = 1 + Phi, A = 1 - gamma Phi
        g_rows.append((N, gap, v, A, gam))
        print(f"      N = {N:.2f}: gap/m = {gap:.4f}, v/c' = {v:.4f}, A = {A:.4f} (1/N = {1/N:.4f}), gamma = {gam:.4f}")
    out['A2'] = np.array(g_rows)

    # A3: second-order price of the stay gate: an extra background -alpha k^2
    print("   A3  extra background of the stay gate: alpha = -[(E(k) - E(0)) - Dirac]/k^2 (k_y = 0.01), "
          "minus the walker's own alpha")

    alpha = lambda Nh, mu: alpha_bg(X, Nh, mu)
    a_rows = []
    for mu in (0.0, 0.03, 0.15):
        own = alpha(ONE, mu)
        for Nh in (0.99, 0.9, 0.75, 0.5, 0.3):
            ex = alpha(Nh, mu) - own
            a_rows.append((mu, Nh, own, ex, ex / (Nh * (1 - Nh))))
            print(f"      mu = {mu:.2f}, N_h = {Nh:.2f}: own alpha {own:+.4f}, extra {ex:+.4f}, "
                  f"extra / (N_h (1 - N_h)) = {ex/(Nh*(1-Nh)):.4f}")
    out['A3'] = np.array(a_rows)


# ═════════════════════════════════════════════════════════════════════════════
def part_B(out):
    print("B. x-chain: transverse acceleration, wait only (1b) vs H2, N(x) = N0 (1 + g x)")
    t0 = time.time()
    X, lam = model_2Ostar()
    Nx, g, SIG, nst = 300, 2e-3, 16.0, 160
    xs = (np.arange(Nx) - Nx // 2) * LAT_A
    ts = np.arange(nst + 1) * DT_HALF
    accel = lambda t, x: 2 * np.polyfit(t, x, 2)[0]
    cases = [('mu=0.03, rest', 0.03, 0.0), ('mu=0.03, k_y=0.1', 0.03, 0.1), ('mu=0.03, k_y=0.2', 0.03, 0.2),
             ('mu=0.15, rest', 0.15, 0.0), ('mu=0.15, k_y=0.3', 0.15, 0.3),
             ('massless, k_y=0.1', 0.0, 0.1), ('massless, k_y=0.2', 0.0, 0.2), ('massless, k_y=0.4', 0.0, 0.4)]
    schemes = {'1b': ('W', lambda N: (N, ONE)), 'H2': ('WH', lambda N: (N, N))}

    def band(kind, Nf, mu, k, N):
        """energy of the selected doublet of the scheme at Bloch k, uniform N."""
        Nw, Nh = Nf(N)
        E = energies(bloch_U(X, kind, k, mu, Nw, Nh))
        kt = np.hypot(k[1], k[2])
        kx = k[0]
        guess = Nw * np.hypot(Nh * CP * np.hypot(kx, kt), 2 * mu) if mu > 0 else Nw * Nh * CP * np.hypot(kx, kt)
        return near(E, guess, 2).mean()

    rows, traj = [], {}
    for cname, mu, ky in cases:
        res = [cname]
        for sname, (kind, Nf) in schemes.items():
            # band group velocity (coordinate) at the base lapse, local velocity in units of the local c
            hk = 1e-4
            vco = (band(kind, Nf, mu, (0, ky + hk, 0), N0) - band(kind, Nf, mu, (0, ky - hk, 0), N0)) / (2 * hk)
            Nw0, Nh0 = Nf(N0)
            vloc = vco / (CP * Nw0 * Nh0)
            # packet from Bloch eigenvectors of the full scheme at the base lapse
            C = expm(1j * X(mu))
            nr = Scheme(kind, None, None).nreg()
            U0 = bloch_U(X, kind, (0, ky, 0), mu, Nw0, Nh0)
            E0, V0 = np.linalg.eig(U0)
            E0 = -np.angle(E0) / DT_HALF
            Eb = band(kind, Nf, mu, (0, ky, 0), N0)
            uref = V0[:, np.argsort(np.abs(E0 - Eb))[0]]
            psi = np.zeros((Nx, 24 * nr), complex)
            for kx in 2 * np.pi * np.fft.fftfreq(Nx) / LAT_A:
                gk = np.exp(-SIG ** 2 * kx ** 2 / 2)
                if gk < 1e-8:
                    continue
                Ek, Vk = np.linalg.eig(bloch_U(X, kind, (kx, ky, 0), mu, Nw0, Nh0))
                Ek = -np.angle(Ek) / DT_HALF
                Q, _ = np.linalg.qr(Vk[:, np.argsort(np.abs(Ek - band(kind, Nf, mu, (kx, ky, 0), N0)))[:2]])
                u = Q @ (Q.conj().T @ uref)
                psi += gk * np.exp(1j * kx * xs)[:, None] * (u / np.linalg.norm(u))[None, :]
            psi /= np.linalg.norm(psi)
            # chain operators
            T = chain_T(C, (ky, 0.0), Nx)
            Cb = sp.kron(sp.eye(Nx), sp.csr_matrix(C), format='csr')
            sc = Scheme(kind, T, Cb)
            Nw, Nh = Nf(N0 * (1 + g * xs))
            sc.set_N(Nw, Nh)
            st = [psi[:, 24 * r:24 * (r + 1)].ravel() for r in range(nr)]
            ph = np.exp(1j * Eb * DT_HALF)
            for _ in range(3):                         # spectral filter against other branches
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

            # semiclassics with the exact band of the scheme
            Hsc = lambda kx, x: band(kind, Nf, mu, (kx, ky, 0), N0 * (1 + g * x))
            kq = np.linspace(-3 / SIG, 3 / SIG, 25)
            wq = np.exp(-SIG ** 2 * kq ** 2)
            hx, hk2 = 1.0, 1e-3
            Fx = np.array([(Hsc(k, hx) - Hsc(k, -hx)) / (2 * hx) for k in kq])
            d2 = np.array([(Hsc(k + hk2, 0) - 2 * Hsc(k, 0) + Hsc(k - hk2, 0)) / hk2 ** 2 for k in kq])
            asc = -np.sum(wq * Fx * d2) / wq.sum()
            res += [asim, asc, vloc]
            print(f"   {cname:18s} {sname}: a_x = {asim:+.3e} (semicl. {asc:+.3e}), v_loc/c = {vloc:.3f}  "
                  f"[{time.time()-t0:.0f}s]")
        a1, s1, _, a2, s2, vl = res[1:]
        R, Rs = (a2 / N0 ** 2) / a1, (s2 / N0 ** 2) / s1

        # ideal metric g00 = -N^2, g_ii = A^2, Dirac dispersion, same packet.  1b: A = 1;
        # family with the H2 base A0 = 1/N0 and a gradient part A = A0 (N/N0)^-gamma (gamma = 1: A = 1/N)
        own = alpha_bg(X, ONE, mu)
        c3 = (alpha_bg(X, 0.75, mu) - own) / (0.75 * 0.25)

        def a_ideal(gam, base=True, art=False):
            def H(kx, x):
                Nl = N0 * (1 + g * x)
                A = (1 / N0) * (Nl / N0) ** (-gam) if base else 1.0
                bg = (own + (c3 * Nl * (1 - Nl) if base else 0.0)) * (kx ** 2 + ky ** 2) if art else 0.0
                return Nl * (np.hypot(2 * mu, CP * np.hypot(kx, ky) / A) - bg)
            Fx = np.array([(H(k, 1.0) - H(k, -1.0)) / 2 for k in kq])
            d2 = np.array([(H(k + 1e-3, 0) - 2 * H(k, 0) + H(k - 1e-3, 0)) / 1e-6 for k in kq])
            return -np.sum(wq * Fx * d2) / wq.sum()
        gg = np.linspace(-1.0, 3.0, 81)
        Rid = np.array([(a_ideal(x) / N0 ** 2) / a_ideal(0.0, base=False) for x in gg])
        R_gr = np.interp(1.0, gg, Rid)
        sens = (np.interp(1.0, gg, Rid) - np.interp(0.0, gg, Rid))
        mono = np.all(np.diff(Rid) > 0) or np.all(np.diff(Rid) < 0)
        o = np.argsort(Rid)
        fit = lambda r: np.interp(r, Rid[o], gg[o]) if (abs(sens) > 0.1 and mono and Rid.min() <= r <= Rid.max()) else np.nan
        gfit, gband = fit(R), fit(Rs)
        # same with the measured stay-gate background alpha = own + c3 N (1 - N)
        Ria = np.array([(a_ideal(x, art=True) / N0 ** 2) / a_ideal(0.0, base=False, art=True) for x in gg])
        R_art = np.interp(1.0, gg, Ria)
        oa = np.argsort(Ria)
        ga = np.interp(R, Ria[oa], gg[oa]) if abs(sens) > 0.1 and Ria.min() <= R <= Ria.max() else np.nan
        res += [R, Rs, R_gr, gfit, gband, 1 + vl ** 2, R_art, ga]
        print(f"      (a_H2 / N0^2) / a_1b: sim {R:.3f}, lattice semicl. {Rs:.3f}; ideal metric gamma = 1: {R_gr:.3f} "
              f"(gamma = 0: {np.interp(0.0, gg, Rid):.3f})" + (f"; fitted gamma: sim {gfit:.3f}, band {gband:.3f}"
                                                               if np.isfinite(gfit) else "; no fit"))
        print(f"      gamma = 1 plus measured background (alpha_own = {own:.4f}, extra {c3:.3f} N(1-N)): {R_art:.3f}"
              + (f"; fitted gamma with background: {ga:.3f}" if np.isfinite(ga) else ""))
        rows.append(res)
    out['B_rows'] = np.array(rows, dtype=object)
    out['B_ts'] = ts
    for key, v in traj.items():
        out['B_traj_' + key] = v
    out['B_g'], out['B_N0'] = g, N0


def main():
    parts = [p.upper() for p in sys.argv[1:]] or ['A', 'B']
    path = data_path('phase_bit_s4_open8.npz')
    out = dict(np.load(path, allow_pickle=True)) if os.path.exists(path) else {}
    for p in parts:
        {'A': part_A, 'B': part_B}[p](out)
        np.savez(path, **out)
    print("saved data/phase_bit_s4_open8.npz")


if __name__ == '__main__':
    main()
